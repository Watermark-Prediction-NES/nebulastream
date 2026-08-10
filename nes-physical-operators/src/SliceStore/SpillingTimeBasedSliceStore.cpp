/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <SliceStore/SpillingTimeBasedSliceStore.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/MemoryPressureSensor.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Time/Timestamp.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

SpillingTimeBasedSliceStore::SpillingTimeBasedSliceStore(
    std::unique_ptr<WindowSlicesStoreInterface> inner_,
    std::unique_ptr<SpillPolicy> policy_,
    std::shared_ptr<StorageBackend> backend_,
    std::unique_ptr<MemoryPressureSensor> sensor_,
    AbstractBufferProvider& buffers_,
    SliceStateSerializer& serializer_)
    : inner(std::move(inner_))
    , spillPolicy(std::move(policy_))
    , backend(std::move(backend_))
    , sensor(std::move(sensor_))
    , buffers(buffers_)
    , serializer(serializer_)
{
    wallClockNow = []
    {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        return Timestamp{static_cast<Timestamp::Underlying>(ms)};
    };
    statsRegistrationId
        = SpillStatsRegistry::instance().registerStore(SpillStatsRegistry::Registration{.stats = &stats, .policy = spillPolicy.get()});
}

SpillingTimeBasedSliceStore::~SpillingTimeBasedSliceStore()
{
    /// Unregister first: the sampler dereferences `stats` and the policy, both of which die below.
    SpillStatsRegistry::instance().unregisterStore(statsRegistrationId);
    if (backend)
    {
        backend->waitForCompletion(std::nullopt);
    }
}

void SpillingTimeBasedSliceStore::setWallClockSourceForTesting(std::function<Timestamp()> clock)
{
    wallClockNow = std::move(clock);
}

std::vector<std::shared_ptr<Slice>> SpillingTimeBasedSliceStore::getSlicesOrCreate(
    const Timestamp timestamp, const std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>& createNewSlice)
{
    /// Pure delegation. The decorator deliberately does NOT track what passes through here: the
    /// JIT-compiled build path holds a SliceStoreRef bound to the concrete inner store and calls it
    /// directly, so most slices are never created through this method. GC reads the inner store's
    /// own live set via getLiveSlices() instead, which sees every slice regardless of the caller.
    return inner->getSlicesOrCreate(timestamp, createNewSlice);
}

std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>>
SpillingTimeBasedSliceStore::getTriggerableWindowSlices(Timestamp globalWatermark)
{
    auto windows = inner->getTriggerableWindowSlices(globalWatermark);
    /// Restore any spilled slices before returning them to the caller (probe phase).
    for (auto& slices : windows | std::views::values)
    {
        for (auto& slicePtr : slices)
        {
            if (slicePtr == nullptr)
            {
                continue;
            }
            auto handleCopy = spillHandlesBySliceEnd.withRLock(
                [&](const auto& map) -> std::optional<SpilledSliceHandle>
                {
                    if (const auto it = map.find(slicePtr->getSliceEnd().getRawValue());
                        it != map.end() && it->second.state == HandleState::Spilled)
                    {
                        return it->second.handle;
                    }
                    return std::nullopt;
                });
            if (handleCopy.has_value())
            {
                restoreSliceSynchronous(*slicePtr, *handleCopy);
                spillHandlesBySliceEnd.withWLock([&](auto& map) { map.erase(slicePtr->getSliceEnd().getRawValue()); });
            }
        }
    }
    return windows;
}

std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> SpillingTimeBasedSliceStore::getAllNonTriggeredSlices()
{
    return inner->getAllNonTriggeredSlices();
}

std::vector<std::shared_ptr<Slice>> SpillingTimeBasedSliceStore::getLiveSlices() const
{
    return inner->getLiveSlices();
}

std::optional<std::shared_ptr<Slice>> SpillingTimeBasedSliceStore::getSliceBySliceEnd(SliceEnd sliceEnd)
{
    auto sliceOpt = inner->getSliceBySliceEnd(sliceEnd);
    if (!sliceOpt.has_value() || sliceOpt.value() == nullptr)
    {
        return sliceOpt;
    }
    const auto handleCopy = spillHandlesBySliceEnd.withRLock(
        [&](const auto& map) -> std::optional<SpilledSliceHandle>
        {
            if (const auto it = map.find(sliceEnd.getRawValue()); it != map.end() && it->second.state == HandleState::Spilled)
            {
                return it->second.handle;
            }
            return std::nullopt;
        });
    if (handleCopy.has_value())
    {
        /// Future enhancement: hand the restore future to the WindowBasedOperatorHandler scheduler
        /// so the worker can run other tasks while the I/O is in flight. For v1 we block here.
        restoreSliceSynchronous(*sliceOpt.value(), *handleCopy);
        spillHandlesBySliceEnd.withWLock([&](auto& map) { map.erase(sliceEnd.getRawValue()); });
    }
    return sliceOpt;
}

void SpillingTimeBasedSliceStore::garbageCollectSlicesAndWindows(Timestamp newGlobalWaterMark)
{
    /// One wall-clock sample per tick, shared by observe() and every per-slice context below, so all
    /// decisions in a tick are taken against the same instant.
    const Timestamp now = wallClockNow();
    spillPolicy->observe(now, newGlobalWaterMark);
    const double pressure = sensor->sample();

    /// Ask the inner store what it actually holds. Not getAllNonTriggeredSlices(), which is
    /// destructive (it consumes an input-pipeline termination and marks windows EMITTED_TO_PROBE).
    const std::vector<std::shared_ptr<Slice>> liveSlices = inner->getLiveSlices();

    for (auto& slicePtr : liveSlices)
    {
        const auto sliceEndKey = slicePtr->getSliceEnd().getRawValue();
        /// Skip slices that already have an on-disk handle — prevents double-spill and races with an ongoing restore.
        const bool inFlight = spillHandlesBySliceEnd.withRLock([&](const auto& map) { return map.find(sliceEndKey) != map.end(); });
        if (inFlight)
        {
            continue;
        }
        const SliceSpillContext ctx{
            .sliceEnd = slicePtr->getSliceEnd(),
            /// Wall clock, NOT the event-time watermark: a predictive policy compares this against the
            /// predictor's wall-clock trigger estimate, so the two must share a clock domain.
            .now = now,
            /// The predictor lives inside the policy, so only the policy can fill this in; it stays
            /// INVALID here and PressureSpillPolicy::decide() computes the estimate itself.
            .predictedTriggerWallClock = Timestamp{Timestamp::INVALID_VALUE},
            .residentBytes = serializer.residentBytes(*slicePtr),
            .spilledBytes = 0,
            .windowState = WindowInfoState::WINDOW_FILLING,
        };
        const auto decision = spillPolicy->decide(ctx, pressure);
        if (decision == SpillDecision::Spill)
        {
            auto spilled = spillSliceSynchronous(*slicePtr);
            spillHandlesBySliceEnd.withWLock(
                [&](auto& map) { map[sliceEndKey] = HandleEntry{.state = HandleState::Spilled, .handle = std::move(spilled)}; });
        }
    }

    /// Refresh the gauges once per tick, after this tick's spills have been recorded.
    const auto spilledNow = spillHandlesBySliceEnd.withRLock([](const auto& map) { return map.size(); });
    stats.spilledSlices.store(static_cast<uint64_t>(spilledNow), std::memory_order_relaxed);
    stats.residentSlices.store(
        static_cast<uint64_t>(liveSlices.size() > spilledNow ? liveSlices.size() - spilledNow : 0), std::memory_order_relaxed);

    /// Delegate to inner; inner manages window state machine and erases fully-aged slices.
    inner->garbageCollectSlicesAndWindows(newGlobalWaterMark);

    /// Sweep handle entries for slices the inner has dropped.
    spillHandlesBySliceEnd.withWLock(
        [&](auto& map)
        {
            for (auto it = map.begin(); it != map.end();)
            {
                if (!inner->getSliceBySliceEnd(SliceEnd{it->first}).has_value())
                {
                    for (const auto& key : it->second.handle.keys)
                    {
                        (void)backend->removeAsync(key);
                    }
                    it = map.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        });
}

void SpillingTimeBasedSliceStore::deleteState()
{
    if (backend)
    {
        backend->waitForCompletion(std::nullopt);
    }
    spillHandlesBySliceEnd.withWLock(
        [&](auto& map)
        {
            for (auto& [_, entry] : map)
            {
                for (const auto& key : entry.handle.keys)
                {
                    (void)backend->removeAsync(key);
                }
            }
            map.clear();
        });
    inner->deleteState();
}

void SpillingTimeBasedSliceStore::incrementNumberOfInputPipelines()
{
    inner->incrementNumberOfInputPipelines();
}

uint64_t SpillingTimeBasedSliceStore::getWindowSize() const
{
    return inner->getWindowSize();
}

bool SpillingTimeBasedSliceStore::isSliceSpilled(SliceEnd sliceEnd) const noexcept
{
    return spillHandlesBySliceEnd.withRLock(
        [&](const auto& map)
        {
            if (const auto it = map.find(sliceEnd.getRawValue()); it != map.end())
            {
                return it->second.state == HandleState::Spilled;
            }
            return false;
        });
}

std::size_t SpillingTimeBasedSliceStore::numSpilledSlices() const noexcept
{
    return spillHandlesBySliceEnd.withRLock([](const auto& map) { return map.size(); });
}

void SpillingTimeBasedSliceStore::restoreSliceSynchronous(Slice& slice, const SpilledSliceHandle& handle)
{
    const auto started = std::chrono::steady_clock::now();
    auto fut = serializer.restore(slice, handle, *backend, buffers);
    const auto result = fut.get();
    if (!result.has_value())
    {
        throw CannotDeserialize("SpillingTimeBasedSliceStore: restore failed: {}", result.error().message);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    stats.addRestore(handle.totalBytes, static_cast<uint64_t>(elapsed));
}

SpilledSliceHandle SpillingTimeBasedSliceStore::spillSliceSynchronous(Slice& slice)
{
    const auto started = std::chrono::steady_clock::now();
    auto fut = serializer.spill(slice, *backend, buffers);
    auto result = fut.get();
    if (!result.has_value())
    {
        throw CannotSerialize("SpillingTimeBasedSliceStore: spill failed: {}", result.error().message);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
    stats.addSpill(result.value().totalBytes, static_cast<uint64_t>(elapsed));
    return std::move(result.value());
}

}
