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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
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
#include <Watermark/WatermarkPredictor.hpp>
#include <Watermark/WatermarkPredictorFactory.hpp>
#include <folly/Synchronized.h>
#include <ErrorHandling.hpp>
#include <SliceCacheConfiguration.hpp>
#include <SlicePreallocationConfiguration.hpp>

namespace NES
{
DefaultTimeBasedSliceStore::DefaultTimeBasedSliceStore(
    const uint64_t windowSize,
    const uint64_t windowSlide,
    SliceCacheConfiguration sliceCacheConfiguration,
    SlicePreallocationConfiguration slicePreallocationConfiguration,
    ObtainSliceFn obtainSliceFn,
    PreemptiveSliceCountFn preemptiveSliceCountFn)
    : sliceCacheConfiguration(std::move(sliceCacheConfiguration))
    , slicePreallocationConfiguration(std::move(slicePreallocationConfiguration))
    , obtainSliceFn(obtainSliceFn ? std::move(obtainSliceFn) : makeObtainSliceFn(this->slicePreallocationConfiguration))
    , preemptiveSliceCountFn(
          preemptiveSliceCountFn ? std::move(preemptiveSliceCountFn) : makePreemptiveSliceCountFn(this->slicePreallocationConfiguration))
    , sliceAssigner(windowSize, windowSlide)
    , sequenceNumber(SequenceNumber::INITIAL)
    , numberOfActiveInputPipelines(0)
{
    *preemptivePredictor.wlock() = makeWatermarkPredictor(this->slicePreallocationConfiguration.preemptivePredictor.getValue());
    wallClockNow = []
    {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        return Timestamp{static_cast<Timestamp::Underlying>(ms)};
    };
}

void DefaultTimeBasedSliceStore::setWallClockSourceForTesting(std::function<Timestamp()> clock)
{
    wallClockNow = std::move(clock);
}

namespace
{
std::shared_ptr<Slice>
createOneSlice(const DefaultTimeBasedSliceStore::CreateSlicesFn& createNewSlice, const SliceStart start, const SliceEnd end)
{
    const auto created = createNewSlice(start, end);
    INVARIANT(created.size() == 1, "We assume that only one slice is created per timestamp for our default time-based slice store.");
    return created[0];
}
}

DefaultTimeBasedSliceStore::PreemptiveSliceCountFn
DefaultTimeBasedSliceStore::makePreemptiveSliceCountFn(const SlicePreallocationConfiguration& config)
{
    /// "off"/"none" -> no predictor. Pick the static strategy once here so the create path never
    /// re-checks; the predictor/assigner/clock arguments are then unused.
    if (makeWatermarkPredictor(config.preemptivePredictor.getValue()) == nullptr)
    {
        return [](const folly::Synchronized<std::unique_ptr<WatermarkPredictor>>&,
                  const SliceAssigner&,
                  const std::function<Timestamp()>&,
                  const SliceEnd,
                  const uint64_t slide,
                  const uint64_t horizonMs) { return horizonMs / slide; };
    }

    const auto cap = config.preemptiveMaxSlices.getValue();
    return [cap](
               const folly::Synchronized<std::unique_ptr<WatermarkPredictor>>& predictorSync,
               const SliceAssigner& sliceAssigner,
               const std::function<Timestamp()>& wallClockNow,
               const SliceEnd sliceEnd,
               const uint64_t slide,
               const uint64_t horizonMs) -> uint64_t
    {
        const auto staticCount = horizonMs / slide;
        /// Predictor is non-null here (selected by config), so no per-call null check.
        auto predictor = predictorSync.rlock();
        const auto deadline = wallClockNow().getRawValue() + horizonMs;
        uint64_t count = 0;
        while (count < cap)
        {
            const auto futureTs = Timestamp{sliceEnd.getRawValue() + (count * slide)};
            const auto end = sliceAssigner.getSliceEndTs(futureTs);
            const auto arrival = (*predictor)->predictWallClock(Timestamp{end.getRawValue()});
            if (arrival.getRawValue() == Timestamp::INVALID_VALUE)
            {
                /// Predictor too cold to answer yet: fall back to the static lookahead.
                return count == 0 ? staticCount : count;
            }
            if (arrival.getRawValue() > deadline)
            {
                break;
            }
            ++count;
        }
        return count;
    };
}

DefaultTimeBasedSliceStore::ObtainSliceFn DefaultTimeBasedSliceStore::makeObtainSliceFn(const SlicePreallocationConfiguration& config)
{
    if (config.recyclePoolSize.getValue() == 0)
    {
        /// No recycle pool: always create fresh; the pool argument is unused.
        return [](folly::Synchronized<std::deque<std::shared_ptr<Slice>>>&,
                  const SliceStart start,
                  const SliceEnd end,
                  const CreateSlicesFn& createNewSlice) { return createOneSlice(createNewSlice, start, end); };
    }

    return [](folly::Synchronized<std::deque<std::shared_ptr<Slice>>>& recyclePool,
              const SliceStart start,
              const SliceEnd end,
              const CreateSlicesFn& createNewSlice) -> std::shared_ptr<Slice>
    {
        auto pooled = recyclePool.withWLock(
            [](auto& pool) -> std::optional<std::shared_ptr<Slice>>
            {
                if (pool.empty())
                {
                    return std::nullopt;
                }
                auto s = std::move(pool.back());
                pool.pop_back();
                return s;
            });
        if (pooled)
        {
            (*pooled)->reset(start, end);
            return std::move(*pooled);
        }
        return createOneSlice(createNewSlice, start, end);
    };
}

DefaultTimeBasedSliceStore::~DefaultTimeBasedSliceStore()
{
    deleteState();
}

std::vector<std::shared_ptr<Slice>> DefaultTimeBasedSliceStore::getSlicesOrCreate(
    const Timestamp timestamp, const std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>& createNewSlice)
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

    auto registerSliceWithWindows
        = [this](
              const std::shared_ptr<Slice>& slice,
              const folly::detail::LockedPtrType<folly::Synchronized<std::map<WindowInfo, SlicesAndState>>>& windowsWriteLocked)
    {
        for (auto windowInfo : sliceAssigner.getAllWindowsForSlice(*slice))
        {
            const auto numberOfExpectedSlices = sliceAssigner.getWindowSize() / sliceAssigner.getWindowSlide();
            const auto [it, success] = windowsWriteLocked->try_emplace(windowInfo, numberOfExpectedSlices);
            if (it->second.windowState == WindowInfoState::EMITTED_TO_PROBE)
            {
                throw WindowingError("We should not add slices to a window that has already been triggered.");
            }
            it->second.windowState = WindowInfoState::WINDOW_FILLING;
            it->second.windowSlices.emplace_back(slice);
        }
    };
    auto newSlice = obtainSliceFn(recyclePool, sliceStart, sliceEnd, createNewSlice);

    /// The current thread has not found a slice, so we need to create one.
    /// It might have happened that another thread acquires the lock before the current thread is finished creating the new slices.
    /// But by not locking the slice store, we reduce the time the current thread holds the lock, increasing the performance.
    /// Therefore, we need to perform another check.
    {
        auto [slicesWriteLocked, windowsWriteLocked] = acquireLocked(slices, windows);
        if (slicesWriteLocked->contains(sliceEnd))
        {
            /// Lost the race; another thread inserted in the gap. Return the recycled slice to the pool
            /// (if recycle is on) so we don't drop it, then return the winning slice.
            if (slicePreallocationConfiguration.recyclePoolSize.getValue() > 0)
            {
                recyclePool.withWLock(
                    [&](auto& pool)
                    {
                        if (pool.size() >= slicePreallocationConfiguration.recyclePoolSize.getValue())
                        {
                            pool.pop_front();
                        }
                        pool.push_back(std::move(newSlice));
                    });
            }
            return {slicesWriteLocked->find(sliceEnd)->second};
        }

        /// At this moment, we can be sure that no slice exists and we can insert the newly created slice into the slice store
        slicesWriteLocked->emplace(sliceEnd, newSlice);
        registerSliceWithWindows(newSlice, windowsWriteLocked);
    }

    /// Preemptive create: while we already hold both locks, also create the upcoming slices the stream
    /// is about to need so future build tuples land on a warm slice instead of racing. The number of
    /// slices is either the static event-time lookahead (no predictor) or predicted from the wall-clock
    /// horizon (with a predictor) — see preemptiveSliceCount.
    const auto horizonMs = slicePreallocationConfiguration.preemptiveCreateHorizonMs.getValue();
    const auto slide = sliceAssigner.getWindowSlide();
    if (horizonMs > 0 and slide > 0)
    {
        for (const auto i : std::views::iota(
                 uint64_t{0}, preemptiveSliceCountFn(preemptivePredictor, sliceAssigner, wallClockNow, sliceEnd, slide, horizonMs)))
        {
            /// sliceEnd is exactly the start of the next slice, so i=0 hits [sliceEnd, sliceEnd+slide).
            const auto futureTs = Timestamp{sliceEnd.getRawValue() + i * slide};
            const auto nextEnd = sliceAssigner.getSliceEndTs(futureTs);
            if (slices.rlock()->contains(nextEnd))
            {
                continue;
            }

            auto preSlice = obtainSliceFn(recyclePool, sliceAssigner.getSliceStartTs(futureTs), nextEnd, createNewSlice);
            auto [slicesWriteLocked, windowsWriteLocked] = acquireLocked(slices, windows);
            if (slicesWriteLocked->contains(nextEnd))
            {
                continue;
            }
            slicesWriteLocked->emplace(nextEnd, preSlice);
            registerSliceWithWindows(preSlice, windowsWriteLocked);
        }
    }

    return {newSlice};
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
    /// GC ticks are our periodic sampling point for the watermark's wall-clock progress: feed the
    /// predictor (event-time = global watermark, wall-clock = now) so the create path can forecast
    /// how far ahead to preallocate. No-op when preemptive_predictor == "off".
    preemptivePredictor.withWLock(
        [&](auto& predictor)
        {
            if (predictor != nullptr)
            {
                predictor->observe(newGlobalWaterMark, wallClockNow());
            }
        });

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
                const auto poolCap = slicePreallocationConfiguration.recyclePoolSize.getValue();
                for (auto slicesLockedIt = slicesWriteLocked->begin(); slicesLockedIt != slicesWriteLocked->end();)
                {
                    const auto& [sliceEnd, slicePtr] = *slicesLockedIt;
                    if (sliceEnd + sliceAssigner.getWindowSize() < newGlobalWaterMark)
                    {
                        /// Only recycle if the slices map is the sole owner. If a window still holds
                        /// the slice (e.g. WAITING_ON_TERMINATION at query end), mutating it via
                        /// reset() would race with that window's eventual probe — fall through to
                        /// the delete path in that case.
                        if (poolCap > 0 and slicePtr.use_count() == 1)
                        {
                            /// Recycle: hand the slice to the pool instead of dropping it. Drop-oldest
                            /// (pop_front) keeps the pool LIFO-warm at the back where obtainSlice pops.
                            recyclePool.withWLock(
                                [&](auto& pool)
                                {
                                    if (pool.size() >= poolCap)
                                    {
                                        pool.pop_front();
                                    }
                                    pool.push_back(slicePtr);
                                });
                            NES_TRACE("Recycled slice with sliceEnd {} into pool", sliceEnd);
                        }
                        else
                        {
                            NES_TRACE("Deleting slice with sliceEnd {} as it is not used anymore", sliceEnd);
                            /// As we are first copying the shared_ptr the destructor of Slice will not be called.
                            /// This allows us to solely collect what slices to delete during holding the lock, while the time-consuming destructor is called without holding any locks
                            slicesToDelete.emplace_back(slicePtr);
                        }
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

    /// Now we can remove/call destructor on every slice without still holding the lock
    slicesToDelete.clear();
}

void DefaultTimeBasedSliceStore::deleteState()
{
    auto [slicesWriteLocked, windowsWriteLocked] = acquireLocked(slices, windows);
    slicesWriteLocked->clear();
    windowsWriteLocked->clear();
    recyclePool.wlock()->clear();
}

void DefaultTimeBasedSliceStore::incrementNumberOfInputPipelines()
{
    numberOfActiveInputPipelines += 1;
}

uint64_t DefaultTimeBasedSliceStore::getWindowSize() const
{
    return sliceAssigner.getWindowSize();
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
    return std::make_unique<DefaultTimeBasedSliceStoreRef>(sliceCacheConfiguration, this, std::move(extractor), std::move(creator));
}

}
