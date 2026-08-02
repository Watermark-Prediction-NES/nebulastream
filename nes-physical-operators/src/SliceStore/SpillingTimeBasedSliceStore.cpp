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
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

SpillingTimeBasedSliceStore::SpillingTimeBasedSliceStore(
    std::unique_ptr<WindowSlicesStoreInterface> inner_,
    std::unique_ptr<SpillPolicy> policy_,
    TierBackends backends_,
    std::unique_ptr<MemoryPressureSensor> sensor_,
    AbstractBufferProvider& buffers_,
    SliceStateSerializer& serializer_)
    : inner(std::move(inner_))
    , spillPolicy(std::move(policy_))
    , tierBackends(std::move(backends_))
    , sensor(std::move(sensor_))
    , buffers(buffers_)
    , serializer(serializer_)
{
    /// Same clock source as DefaultTimeBasedSliceStore — the predictor is shared between them and its
    /// observations must land in one domain.
    wallClockNow = []
    {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        return Timestamp{static_cast<Timestamp::Underlying>(ms)};
    };
}

void SpillingTimeBasedSliceStore::setWallClockSourceForTesting(std::function<Timestamp()> clock)
{
    wallClockNow = std::move(clock);
}

SpillingTimeBasedSliceStore::~SpillingTimeBasedSliceStore()
{
    for (const auto& tierBackend : tierBackends)
    {
        if (tierBackend)
        {
            tierBackend->waitForCompletion(std::nullopt);
        }
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
            if (slicePtr != nullptr)
            {
                restoreIfSpilled(*slicePtr);
            }
        }
    }
    return windows;
}

void SpillingTimeBasedSliceStore::restoreIfSpilled(Slice& slice)
{
    const auto sliceEndKey = slice.getSliceEnd().getRawValue();
    /// The restore runs under the map's write lock, the same lock a GC-tick tier move takes. Every
    /// backend resolves its futures inline, so there is nothing to overlap with, and releasing the lock
    /// around the I/O would open a window in which a concurrent probe finds the slice neither resident
    /// nor readable and silently returns it empty. If backends ever become genuinely async, this needs
    /// a per-slice wait instead of a lock.
    spillHandlesBySliceEnd.withWLock(
        [&](auto& map)
        {
            const auto it = map.find(sliceEndKey);
            if (it == map.end())
            {
                return;
            }
            const auto tier = it->second.tier;
            const auto handle = it->second.handle;
            /// Throws on failure, which leaves the entry in place — the slice stays readable from its tier.
            restoreSliceSynchronous(slice, handle, tier);
            map.erase(it);
            removeHandleObjects(handle, tier);
        });
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
    /// Future enhancement: hand the restore future to the WindowBasedOperatorHandler scheduler
    /// so the worker can run other tasks while the I/O is in flight. For v1 we block here.
    restoreIfSpilled(*sliceOpt.value());
    return sliceOpt;
}

void SpillingTimeBasedSliceStore::garbageCollectSlicesAndWindows(Timestamp newGlobalWaterMark)
{
    const Timestamp now = wallClockNow();
    spillPolicy->observe(now, newGlobalWaterMark);
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
        const auto entry = spillHandlesBySliceEnd.withRLock(
            [&](const auto& map) -> std::optional<HandleEntry>
            {
                const auto it = map.find(sliceEndKey);
                return it != map.end() ? std::optional{it->second} : std::nullopt;
            });
        /// Every slice is re-evaluated, including already-spilled ones. The old code skipped anything
        /// with a handle, which is what made demotion and promotion impossible. applyTierTransition
        /// re-checks the tier under the write lock, so sampling it here without the lock is safe.
        const SliceTier currentTier = entry.has_value() ? entry->tier : SliceTier::Resident;
        const SliceSpillContext ctx{
            .sliceEnd = slicePtr->getSliceEnd(),
            .now = now,
            .currentTier = currentTier,
            .residentBytes = serializer.residentBytes(*slicePtr),
            .spilledBytes = entry.has_value() ? entry->handle.totalBytes : 0,
            .windowState = WindowInfoState::WINDOW_FILLING,
        };
        if (const auto target = spillPolicy->decide(ctx, pressure); target != currentTier)
        {
            applyTierTransition(*slicePtr, sliceEndKey, currentTier, target);
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
                    if (auto* tierBackend = backendFor(it->second.tier))
                    {
                        for (const auto& key : it->second.handle.keys)
                        {
                            (void)tierBackend->removeAsync(key);
                        }
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
    for (const auto& tierBackend : tierBackends)
    {
        if (tierBackend)
        {
            tierBackend->waitForCompletion(std::nullopt);
        }
    }
    spillHandlesBySliceEnd.withWLock(
        [&](auto& map)
        {
            for (auto& [_, entry] : map)
            {
                if (auto* tierBackend = backendFor(entry.tier))
                {
                    for (const auto& key : entry.handle.keys)
                    {
                        (void)tierBackend->removeAsync(key);
                    }
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
    return spillHandlesBySliceEnd.withRLock([&](const auto& map) { return map.contains(sliceEnd.getRawValue()); });
}

std::size_t SpillingTimeBasedSliceStore::numSpilledSlices() const noexcept
{
    return spillHandlesBySliceEnd.withRLock([](const auto& map) { return map.size(); });
}

void SpillingTimeBasedSliceStore::restoreSliceSynchronous(Slice& slice, const SpilledSliceHandle& handle, SliceTier from)
{
    auto* fromBackend = backendFor(from);
    INVARIANT(fromBackend != nullptr, "SpillingTimeBasedSliceStore: no backend wired for the tier holding this slice");
    auto fut = serializer.restore(slice, handle, *fromBackend, buffers);
    const auto result = fut.get();
    if (!result.has_value())
    {
        throw CannotDeserialize("SpillingTimeBasedSliceStore: restore failed: {}", result.error().message);
    }
}

SpilledSliceHandle SpillingTimeBasedSliceStore::spillSliceSynchronous(Slice& slice, SliceTier to)
{
    auto* toBackend = backendFor(to);
    INVARIANT(toBackend != nullptr, "SpillingTimeBasedSliceStore: no backend wired for the target tier");
    auto fut = serializer.spill(slice, *toBackend, buffers);
    auto result = fut.get();
    if (!result.has_value())
    {
        throw CannotSerialize("SpillingTimeBasedSliceStore: spill failed: {}", result.error().message);
    }
    return std::move(result.value());
}

void SpillingTimeBasedSliceStore::applyTierTransition(
    Slice& slice, const Timestamp::Underlying sliceEndKey, const SliceTier from, const SliceTier to)
{
    /// A policy may name a tier this query never wired up (e.g. "tiered" horizons on an in-memory-only
    /// setup). Leaving the slice where it is is always safe.
    if (to != SliceTier::Resident && backendFor(to) == nullptr)
    {
        return;
    }

    spillHandlesBySliceEnd.withWLock(
        [&](auto& map)
        {
            const auto it = map.find(sliceEndKey);
            /// Re-validate under the lock: the GC loop sampled the tier without holding it, so a probe
            /// may have restored this slice in the meantime.
            const SliceTier currentTier = it != map.end() ? it->second.tier : SliceTier::Resident;
            if (currentTier != from || currentTier == to)
            {
                return;
            }

            if (from == SliceTier::Resident)
            {
                map[sliceEndKey] = HandleEntry{.tier = to, .handle = spillSliceSynchronous(slice, to)};
                return;
            }

            const auto handle = it->second.handle;
            if (to == SliceTier::Resident)
            {
                /// Eager promotion: pull the slice back before the probe asks for it.
                restoreSliceSynchronous(slice, handle, from);
                map.erase(it);
                removeHandleObjects(handle, from);
                return;
            }

            /// Lateral move: copy the stored bytes across backends. Going through the serializer instead
            /// would round-trip every page through the buffer pool for no reason.
            for (const auto& key : handle.keys)
            {
                if (auto copied = copySpillObject(*backendFor(from), *backendFor(to), key); !copied.has_value())
                {
                    NES_WARNING(
                        "SpillingTimeBasedSliceStore: tier move {} -> {} failed for slice {}: {}; leaving the slice in place",
                        static_cast<uint32_t>(from),
                        static_cast<uint32_t>(to),
                        sliceEndKey,
                        copied.error().message);
                    /// Drop whatever reached the destination. The entry still names `from`, so the slice
                    /// stays readable from the tier that has always held it.
                    removeHandleObjects(handle, to);
                    return;
                }
            }
            it->second.tier = to;
            removeHandleObjects(handle, from);
        });
}

void SpillingTimeBasedSliceStore::removeHandleObjects(const SpilledSliceHandle& handle, const SliceTier tier)
{
    if (auto* tierBackend = backendFor(tier))
    {
        for (const auto& key : handle.keys)
        {
            (void)tierBackend->removeAsync(key);
        }
    }
}

SliceTier SpillingTimeBasedSliceStore::tierOf(SliceEnd sliceEnd) const noexcept
{
    return spillHandlesBySliceEnd.withRLock(
        [&](const auto& map)
        {
            const auto it = map.find(sliceEnd.getRawValue());
            return it != map.end() ? it->second.tier : SliceTier::Resident;
        });
}

}
