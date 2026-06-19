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
}

SpillingTimeBasedSliceStore::~SpillingTimeBasedSliceStore()
{
    if (backend)
    {
        backend->waitForCompletion(std::nullopt);
    }
}

std::vector<std::shared_ptr<Slice>> SpillingTimeBasedSliceStore::getSlicesOrCreate(
    const Timestamp timestamp, const std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>& createNewSlice)
{
    auto slices = inner->getSlicesOrCreate(timestamp, createNewSlice);
    /// Track slices weakly so the decorator can iterate them during GC ticks without consuming
    /// the inner store's destructive pipeline counter on each call.
    observedSlices.withWLock(
        [&](auto& map)
        {
            for (const auto& slice : slices)
            {
                if (slice != nullptr)
                {
                    map[slice->getSliceEnd().getRawValue()] = slice;
                }
            }
        });
    return slices;
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
    spillPolicy->observe(newGlobalWaterMark, newGlobalWaterMark);
    const double pressure = sensor->sample();

    /// Snapshot live slices from our weak tracking map. We never call the inner store's
    /// destructive getAllNonTriggeredSlices here.
    std::vector<std::shared_ptr<Slice>> liveSlices;
    observedSlices.withWLock(
        [&](auto& map)
        {
            for (auto it = map.begin(); it != map.end();)
            {
                if (auto strong = it->second.lock())
                {
                    liveSlices.push_back(std::move(strong));
                    ++it;
                }
                else
                {
                    it = map.erase(it);
                }
            }
        });

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
            .now = newGlobalWaterMark,
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
    auto fut = serializer.restore(slice, handle, *backend, buffers);
    const auto result = fut.get();
    if (!result.has_value())
    {
        throw CannotDeserialize("SpillingTimeBasedSliceStore: restore failed: {}", result.error().message);
    }
}

SpilledSliceHandle SpillingTimeBasedSliceStore::spillSliceSynchronous(Slice& slice)
{
    auto fut = serializer.spill(slice, *backend, buffers);
    auto result = fut.get();
    if (!result.has_value())
    {
        throw CannotSerialize("SpillingTimeBasedSliceStore: spill failed: {}", result.error().message);
    }
    return std::move(result.value());
}

}
