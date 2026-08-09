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

#include <SliceStore/DefaultTimeBasedSliceStore.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/DefaultTimeBasedSliceStoreRef.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/SliceAssigner.hpp>
#include <SliceStore/SliceStoreRef.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/Logger.hpp>
#include <folly/Synchronized.h>
#include <ErrorHandling.hpp>
#include <SliceCacheConfiguration.hpp>
#include <SliceStoreConfiguration.hpp>

namespace NES
{
DefaultTimeBasedSliceStore::DefaultTimeBasedSliceStore(
    const uint64_t windowSize,
    const uint64_t windowSlide,
    SliceCacheConfiguration sliceCacheConfiguration,
    SliceStoreConfiguration sliceStoreConfiguration)
    /// A configured capacity of 0 means "as many slices as are alive in the steady state".
    : WindowSlicesStoreInterface{
          sliceStoreConfiguration.slicePoolCapacity.getValue() != 0 ? sliceStoreConfiguration.slicePoolCapacity.getValue()
                                                                    : std::max<uint64_t>(windowSize / windowSlide, 1),
          sliceStoreConfiguration.enableSliceRecycling.getValue()}
    , sliceCacheConfiguration(std::move(sliceCacheConfiguration))
    , sliceStoreConfiguration(std::move(sliceStoreConfiguration))
    , sliceAssigner(windowSize, windowSlide)
    , sequenceNumber(SequenceNumber::INITIAL)
    , numberOfActiveInputPipelines(0)
{
}

void DefaultTimeBasedSliceStore::poolRetiredSlice(std::shared_ptr<Slice> slice, const bool pristine)
{
    /// A slice that is still referenced elsewhere (e.g. a trigger buffer in flight) must not get a new
    /// tenant; it simply dies the normal way once the last reference drops.
    if (not sliceRecyclingEnabled or slice.use_count() != 1)
    {
        return;
    }
    if (not pristine and not slice->resetForReuse())
    {
        return;
    }
    if (const auto pool = slicePool.wlock(); pool->size() < slicePoolCapacity)
    {
        pool->push_back(std::move(slice));
    }
}

DefaultTimeBasedSliceStore::~DefaultTimeBasedSliceStore()
{
    deleteState();
}

std::vector<std::shared_ptr<Slice>>
DefaultTimeBasedSliceStore::getSlicesOrCreate(const Timestamp timestamp, const SliceCreateFunction& createNewSlice)
{
    /// We first check, if the slice already exist in the slice store
    const auto sliceStart = sliceAssigner.getSliceStartTs(timestamp);
    const auto sliceEnd = sliceAssigner.getSliceEndTs(timestamp);
    {
        const auto slicesWriteLocked = slices.rlock();
        if (const auto existingSlice = slicesWriteLocked->find(sliceEnd); existingSlice != slicesWriteLocked->end())
        {
            return {existingSlice->second};
        }
    }

    /// Offer the create function a retired slice to reuse; it decides whether the candidate fits and
    /// simply drops it otherwise.
    std::shared_ptr<Slice> recycledCandidate;
    if (sliceRecyclingEnabled)
    {
        if (const auto pool = slicePool.wlock(); not pool->empty())
        {
            recycledCandidate = std::move(pool->back());
            pool->pop_back();
        }
    }

    /// The current thread has not found a slice, so we need to create one.
    /// It might have happened that another thread acquires the lock before the current thread is finished creating the new slices.
    /// But by not locking the slice store, we reduce the time the current thread holds the lock, increasing the performance.
    /// Therefore, we need to perform another check.
    /// The creation is timed outside of any lock scope, so the measurement never inflates lock hold times.
    const auto creationStart = std::chrono::steady_clock::now();
    auto newSlices = createNewSlice(sliceStart, sliceEnd, recycledCandidate);
    const auto creationDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - creationStart);
    createdSlices.fetch_add(1, std::memory_order_relaxed);
    sliceCreationNanos.fetch_add(creationDuration.count(), std::memory_order_relaxed);
    /// Racy read-modify-write is fine: this is a smoothed hint for group sizing, not an exact counter.
    const auto previousEwma = ewmaCreationNanos.load(std::memory_order_relaxed);
    const auto sample = static_cast<uint64_t>(creationDuration.count());
    ewmaCreationNanos.store(previousEwma == 0 ? sample : (previousEwma * 7 + sample) / 8, std::memory_order_relaxed);
    INVARIANT(newSlices.size() == 1, "We assume that only one slice is created per timestamp for our default time-based slice store.");
    auto [slicesWriteLocked, windowsWriteLocked] = acquireLocked(slices, windows);
    if (slicesWriteLocked->contains(sliceEnd))
    {
        wastedSliceCreations.fetch_add(1, std::memory_order_relaxed);
        /// The loser's freshly built (or freshly reused) slice never became visible to anyone: pool it as
        /// pristine inventory. Do NOT fire onSliceRetired here — its slice end is the winner's live slice.
        const auto winner = slicesWriteLocked->find(sliceEnd)->second;
        slicesWriteLocked.unlock();
        windowsWriteLocked.unlock();
        /// The recycled candidate may still hold the same slice; drop it first so the pool sees the only reference.
        recycledCandidate.reset();
        poolRetiredSlice(std::move(newSlices[0]), true);
        return {winner};
    }

    /// At this moment, we can be sure that no slice exists and we can insert the newly created slice into the slice store
    auto newSlice = newSlices[0];
    slicesWriteLocked->emplace(sliceEnd, newSlice);
    slicesWriteLocked.unlock();

    /// Update the state of all windows that contain this slice as we have to expect new tuples
    for (auto windowInfo : sliceAssigner.getAllWindowsForSlice(*newSlice))
    {
        const auto numberOfExpectedSlices = sliceAssigner.getWindowSize() / sliceAssigner.getWindowSlide();
        const auto [it, success] = windowsWriteLocked->try_emplace(windowInfo, numberOfExpectedSlices);
        if (it->second.windowState == WindowInfoState::EMITTED_TO_PROBE)
        {
            throw WindowingError("We should not add slices to a window that has already been triggered.");
        }
        it->second.windowState = WindowInfoState::WINDOW_FILLING;
        it->second.windowSlices.emplace_back(newSlice);
    }

    return {newSlice};
}

uint64_t DefaultTimeBasedSliceStore::claimOrDeferSliceRange(
    const Timestamp minTs, const Timestamp maxTs, const double watermarkRate, const SliceCreateFunction& createNewSlice)
{
    PRECONDITION(minTs <= maxTs, "Range start {} must not exceed range end {}", minTs, maxTs);

    /// Clamp to the collector's horizon. An out-of-order buffer's minTs can sit below it, and recreating a
    /// slice there re-emplaces its collected window in WINDOW_FILLING, past the EMITTED_TO_PROBE guard: the
    /// window triggers twice and the probe gets a slice the next collection frees underneath it.
    /// Collection retires a slice once sliceEnd + windowSize < watermark, so starting at
    /// gcWatermark - windowSize keeps every slice created here above the horizon.
    auto claimMinTs = minTs;
    if (const auto gcWatermark = lastGcWatermark.load(std::memory_order_relaxed); gcWatermark > sliceAssigner.getWindowSize())
    {
        if (const auto oldestLiveSliceEnd = gcWatermark - sliceAssigner.getWindowSize(); claimMinTs.getRawValue() < oldestLiveSliceEnd)
        {
            claimMinTs = Timestamp(oldestLiveSliceEnd);
        }
    }
    if (claimMinTs > maxTs)
    {
        /// The whole buffer is behind the horizon. Its records are late; per-tuple creation deals with them.
        return 0;
    }

    /// A buffer whose range spans absurdly many slices (a symptom of unstamped or corrupt metadata, not
    /// of real data) must not send us iterating slice by slice over it; per-tuple creation handles it.
    const auto spanSlices = ((maxTs.getRawValue() - claimMinTs.getRawValue()) / sliceAssigner.getWindowSlide()) + 1;
    if (spanSlices > 10 * sliceStoreConfiguration.maxSliceGroupSize.getValue())
    {
        return 0;
    }

    /// Fast path: one read lock to see whether the buffer's range is already fully covered.
    uint64_t missingSlices = 0;
    {
        const auto slicesReadLocked = slices.rlock();
        for (auto ts = claimMinTs; ts <= maxTs; ts = sliceAssigner.getSliceEndTs(ts))
        {
            if (not slicesReadLocked->contains(sliceAssigner.getSliceEndTs(ts)))
            {
                ++missingSlices;
            }
        }
    }
    if (missingSlices == 0)
    {
        return 0;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto creationCostNanos = ewmaCreationNanos.load(std::memory_order_relaxed);
    const auto groupSize = computeSliceGroupSize(
        missingSlices,
        watermarkRate,
        creationCostNanos,
        sliceAssigner.getWindowSlide(),
        sliceStoreConfiguration.maxSliceGroupSize.getValue());
    const auto claimMaxTs = Timestamp(maxTs.getRawValue() + ((groupSize - missingSlices) * sliceAssigner.getWindowSlide()));

    uint64_t claimId = 0;
    {
        const auto rangesWriteLocked = inFlightRanges.wlock();
        for (const auto& range : *rangesWriteLocked)
        {
            if (range.rangeStart <= maxTs and claimMinTs <= range.rangeEnd)
            {
                /// Another thread is already creating an overlapping range: retry roughly when it is done.
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(range.expectedCompletion - now).count();
                return static_cast<uint64_t>(std::max<int64_t>(remaining, 1));
            }
        }
        claimId = nextClaimId.fetch_add(1, std::memory_order_relaxed);
        rangesWriteLocked->push_back(InFlightRange{
            .claimId = claimId,
            .rangeStart = claimMinTs,
            .rangeEnd = claimMaxTs,
            .expectedCompletion = now + std::chrono::nanoseconds{groupSize * creationCostNanos}});
    }

    /// The claim must clear whatever happens below; a stuck claim would make every deferring thread
    /// retry forever and block a graceful query stop.
    struct ClaimReleaser
    {
        DefaultTimeBasedSliceStore& store;
        uint64_t claimId;

        ~ClaimReleaser()
        {
            std::erase_if(*store.inFlightRanges.wlock(), [&](const auto& range) { return range.claimId == claimId; });
        }
    };

    const ClaimReleaser releaseClaim{.store = *this, .claimId = claimId};

    for (auto ts = claimMinTs; ts <= claimMaxTs; ts = sliceAssigner.getSliceEndTs(ts))
    {
        std::ignore = getSlicesOrCreate(ts, createNewSlice);
    }
    return 0;
}

std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>>
DefaultTimeBasedSliceStore::getTriggerableWindowSlices(const Timestamp globalWatermark)
{
    /// For performance reasons, we check if we can acquire a lock and if not we then simply skip checking if we can trigger anything
    const auto windowsWriteLocked = windows.tryWLock();
    if (windowsWriteLocked.isNull())
    {
        return {};
    }

    /// We are iterating over all windows and check if they can be triggered
    /// A window can be triggered if all sides have been filled and the window end is smaller than the new global watermark
    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> windowsToSlices;
    for (auto& [windowInfo, windowSlicesAndState] : *windowsWriteLocked)
    {
        if (windowInfo.windowEnd >= globalWatermark)
        {
            /// As the windows are sorted (due to std::map), we can break here as we will not find any windows with a smaller window end
            break;
        }
        if (windowSlicesAndState.windowState == WindowInfoState::EMITTED_TO_PROBE)
        {
            /// This window has already been triggered
            continue;
        }

        windowSlicesAndState.windowState = WindowInfoState::EMITTED_TO_PROBE;
        /// As the windows are sorted, we can simply increment the sequence number here.
        const auto newSequenceNumber = SequenceNumber(sequenceNumber++);
        for (auto& slice : windowSlicesAndState.windowSlices)
        {
            windowsToSlices[{windowInfo, newSequenceNumber}].emplace_back(slice);
        }
    }
    return windowsToSlices;
}

std::optional<std::shared_ptr<Slice>> DefaultTimeBasedSliceStore::getSliceBySliceEnd(const SliceEnd sliceEnd)
{
    if (const auto slicesReadLocked = slices.rlock(); slicesReadLocked->contains(sliceEnd))
    {
        return slicesReadLocked->find(sliceEnd)->second;
    }
    return {};
}

std::vector<std::shared_ptr<Slice>> DefaultTimeBasedSliceStore::getAllSlices()
{
    const auto slicesReadLocked = slices.rlock();
    std::vector<std::shared_ptr<Slice>> allSlices;
    allSlices.reserve(slicesReadLocked->size());
    for (const auto& slicePtr : *slicesReadLocked | std::views::values)
    {
        allSlices.push_back(slicePtr);
    }
    return allSlices;
}

std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> DefaultTimeBasedSliceStore::getAllNonTriggeredSlices()
{
    /// Acquiring a lock for the windows, as we have to iterate over all windows and trigger all non-triggered windows
    const auto windowsWriteLocked = windows.wlock();

    /// numberOfActiveInputPipelines is guarded by the windows lock.
    /// If this method gets called, we know that an input pipeline has terminated.
    INVARIANT(numberOfActiveInputPipelines > 0, "Method should not be called if all input pipelines have terminated.");
    numberOfActiveInputPipelines -= 1;

    /// Creating a lambda to add all slices to the return map windowsToSlices
    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> windowsToSlices;
    auto addAllSlicesToReturnMap = [&windowsToSlices, this](const WindowInfo& windowInfo, SlicesAndState& windowSlicesAndState)
    {
        const auto newSequenceNumber = SequenceNumber(sequenceNumber++);
        for (auto& slice : windowSlicesAndState.windowSlices)
        {
            windowsToSlices[{windowInfo, newSequenceNumber}].emplace_back(slice);
        }
        windowSlicesAndState.windowState = WindowInfoState::EMITTED_TO_PROBE;
    };

    /// We are iterating over all windows and check if they can be triggered
    for (auto& [windowInfo, windowSlicesAndState] : *windowsWriteLocked)
    {
        switch (windowSlicesAndState.windowState)
        {
            case WindowInfoState::EMITTED_TO_PROBE:
                continue;
            case WindowInfoState::WINDOW_FILLING: {
                /// If we are waiting on another pipeline to terminate, we can not trigger the window yet
                if (numberOfActiveInputPipelines > 0)
                {
                    windowSlicesAndState.windowState = WindowInfoState::WAITING_ON_TERMINATION;
                    NES_TRACE(
                        "Waiting on termination for window end {} and number of active input pipelines {}",
                        windowInfo.windowEnd,
                        numberOfActiveInputPipelines);
                    break;
                }
                addAllSlicesToReturnMap(windowInfo, windowSlicesAndState);
                break;
            }
            case WindowInfoState::WAITING_ON_TERMINATION: {
                /// Checking if all input pipelines have terminated (i.e., the number of active input pipelines is 0, as we will decrement it during fetch_sub)
                NES_TRACE(
                    "Checking if all input pipelines have terminated for window with window end {} and number of active pipelines {}",
                    windowInfo.windowEnd,
                    numberOfActiveInputPipelines);
                if (numberOfActiveInputPipelines > 0)
                {
                    continue;
                }
                addAllSlicesToReturnMap(windowInfo, windowSlicesAndState);
                break;
            }
        }
    }

    return windowsToSlices;
}

void DefaultTimeBasedSliceStore::garbageCollectSlicesAndWindows(const Timestamp newGlobalWaterMark)
{
    /// Publish before collecting, so a concurrent claim clamps against this pass rather than the previous
    /// one. Threads collect at different watermarks, hence the max: the horizon must never move backwards.
    {
        auto previousHorizon = lastGcWatermark.load(std::memory_order_relaxed);
        while (previousHorizon < newGlobalWaterMark.getRawValue()
               and not lastGcWatermark.compare_exchange_weak(previousHorizon, newGlobalWaterMark.getRawValue(), std::memory_order_relaxed))
        {
        }
    }

    std::vector<std::shared_ptr<Slice>> slicesToDelete;
    {
        NES_TRACE("Performing garbage collection for new global watermark {}", newGlobalWaterMark);

        {
            /// Solely acquiring a lock for the windows
            if (const auto windowsWriteLocked = windows.tryWLock())
            {
                /// 1. We iterate over all windows and erase them if they can be deleted
                /// This condition is true, if the window end is smaller than the new global watermark of the probe phase.
                for (auto windowsLockedIt = windowsWriteLocked->cbegin(); windowsLockedIt != windowsWriteLocked->cend();)
                {
                    const auto& [windowInfo, windowSlicesAndState] = *windowsLockedIt;
                    if (windowInfo.windowEnd < newGlobalWaterMark and windowSlicesAndState.windowState == WindowInfoState::EMITTED_TO_PROBE)
                    {
                        windowsLockedIt = windowsWriteLocked->erase(windowsLockedIt);
                    }
                    else if (windowInfo.windowEnd > newGlobalWaterMark)
                    {
                        /// As the windows are sorted (due to std::map), we can break here as we will not find any windows with a smaller window end
                        break;
                    }
                    else
                    {
                        ++windowsLockedIt;
                    }
                }
            }
        }

        {
            /// Solely acquiring a lock for the slices
            if (const auto slicesWriteLocked = slices.tryWLock())
            {
                /// 2. We gather all slices if they are not used in any window that has not been triggered/can not be deleted yet
                for (auto slicesLockedIt = slicesWriteLocked->begin(); slicesLockedIt != slicesWriteLocked->end();)
                {
                    const auto& [sliceEnd, slicePtr] = *slicesLockedIt;
                    if (sliceEnd + sliceAssigner.getWindowSize() < newGlobalWaterMark)
                    {
                        NES_TRACE("Deleting slice with sliceEnd {} as it is not used anymore", sliceEnd);
                        /// As we are first copying the shared_ptr the destructor of Slice will not be called.
                        /// This allows us to solely collect what slices to delete during holding the lock, while the time-consuming destructor is called without holding any locks
                        slicesToDelete.emplace_back(slicePtr);
                        slicesLockedIt = slicesWriteLocked->erase(slicesLockedIt);
                    }
                    else
                    {
                        /// As the slices are sorted (due to std::map), we can break here as we will not find any slices with a smaller slice end
                        break;
                    }
                }
            }
        }
    }

    /// Every gathered slice's identity dies here. Outside all locks: the retirement callback and the
    /// data structure reset are the time-consuming parts, and any slice not pooled destructs right away.
    for (auto& slice : slicesToDelete)
    {
        if (onSliceRetired)
        {
            onSliceRetired(slice->getSliceEnd());
        }
        poolRetiredSlice(std::move(slice), false);
    }
}

void DefaultTimeBasedSliceStore::deleteState()
{
    auto [slicesWriteLocked, windowsWriteLocked] = acquireLocked(slices, windows);
    slicesWriteLocked->clear();
    windowsWriteLocked->clear();
    slicePool.wlock()->clear();
}

void DefaultTimeBasedSliceStore::incrementNumberOfInputPipelines()
{
    numberOfActiveInputPipelines += 1;
}

uint64_t DefaultTimeBasedSliceStore::getWindowSize() const
{
    return sliceAssigner.getWindowSize();
}

SliceStoreStatistics DefaultTimeBasedSliceStore::getStatistics() const
{
    return {
        .createdSlices = createdSlices.load(std::memory_order_relaxed),
        .wastedSliceCreations = wastedSliceCreations.load(std::memory_order_relaxed),
        .sliceCreationNanos = sliceCreationNanos.load(std::memory_order_relaxed)};
}

std::span<std::byte> DefaultTimeBasedSliceStore::allocateSpaceForSliceCache(
    uint64_t sliceCacheMemorySize, PipelineId pipelineId, AbstractBufferProvider& bufferProvider)
{
    INVARIANT(not pipelineIdToSliceCacheStarts.rlock()->contains(pipelineId), "We expect this method to be called once per pipelineId!");

    auto buffer = bufferProvider.getUnpooledBuffer(sliceCacheMemorySize);
    if (not buffer.has_value())
    {
        throw BufferAllocationFailure("Can not allocate buffer for slice cache of size {}", sliceCacheMemorySize);
    }

    /// We set everything to 0, as there might be old data in the tuple buffer
    std::ranges::fill(buffer.value().getAvailableMemoryArea(), std::byte{0});
    auto sliceCacheStartBuffer = std::make_unique<TupleBuffer>(buffer.value());
    const auto& memArea = sliceCacheStartBuffer->getAvailableMemoryArea();
    pipelineIdToSliceCacheStarts.wlock()->emplace(pipelineId, std::move(sliceCacheStartBuffer));
    return memArea;
}

std::unique_ptr<SliceStoreRef> DefaultTimeBasedSliceStore::createSliceStoreRef(
    DefaultTimeBasedSliceStoreRef::DataStructureExtractor extractor, DefaultTimeBasedSliceStoreRef::CreateSlicesFunction creator)
{
    return std::make_unique<DefaultTimeBasedSliceStoreRef>(
        sliceCacheConfiguration, sliceStoreConfiguration, this, std::move(extractor), std::move(creator));
}

}
