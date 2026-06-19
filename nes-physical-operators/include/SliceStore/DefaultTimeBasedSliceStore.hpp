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

#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <SliceStore/DefaultTimeBasedSliceStoreRef.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <folly/Synchronized.h>

#include <Identifiers/Identifiers.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/SliceAssigner.hpp>
#include <SliceStore/SliceStoreRef.hpp>
#include <Time/Timestamp.hpp>
#include <SliceCacheConfiguration.hpp>
#include <SlicePreallocationConfiguration.hpp>

namespace NES
{

class WatermarkPredictor;

/// This struct stores a slice ptr and the state. We require this information, as we have to know the state of a slice for a given window
struct SlicesAndState
{
    explicit SlicesAndState(const uint64_t numberOfExpectedSlices) : windowState(WindowInfoState::WINDOW_FILLING)
    {
        windowSlices.reserve(numberOfExpectedSlices);
    }

    std::vector<std::shared_ptr<Slice>> windowSlices;
    WindowInfoState windowState;
};

class DefaultTimeBasedSliceStore final : public WindowSlicesStoreInterface
{
public:
    using CreateSlicesFn = std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>;

    /// How a build tuple gets its slice: reuse a recycled one or always create fresh. The recycle pool
    /// is a store member, so it is passed in as an argument rather than captured — the closure only
    /// encodes the strategy, picked once at lowering so the per-call recycle on/off branch is gone.
    using ObtainSliceFn = std::function<std::shared_ptr<Slice>(
        folly::Synchronized<std::deque<std::shared_ptr<Slice>>>& recyclePool, SliceStart, SliceEnd, const CreateSlicesFn&)>;

    /// How many slices to preemptively create ahead of a new slice: static event-time lookahead or
    /// predictor-driven. Predictor/assigner/clock are store-owned (and the clock is test-overridable),
    /// so they are passed in as arguments; lowering picks the strategy, removing the per-call predictor
    /// null check.
    using PreemptiveSliceCountFn = std::function<uint64_t(
        const folly::Synchronized<std::unique_ptr<WatermarkPredictor>>& predictor,
        const SliceAssigner& sliceAssigner,
        const std::function<Timestamp()>& wallClockNow,
        SliceEnd sliceEnd,
        uint64_t slide,
        uint64_t horizonMs)>;

    /// Build the two strategies from config. Called by the lowering and passed into the constructor;
    /// the constructor also falls back to these when a strategy is not supplied.
    static ObtainSliceFn makeObtainSliceFn(const SlicePreallocationConfiguration& config);
    static PreemptiveSliceCountFn makePreemptiveSliceCountFn(const SlicePreallocationConfiguration& config);

    DefaultTimeBasedSliceStore(
        uint64_t windowSize,
        uint64_t windowSlide,
        SliceCacheConfiguration sliceCacheConfiguration,
        SlicePreallocationConfiguration slicePreallocationConfiguration = {},
        ObtainSliceFn obtainSliceFn = nullptr,
        PreemptiveSliceCountFn preemptiveSliceCountFn = nullptr);

    ~DefaultTimeBasedSliceStore() override;
    std::vector<std::shared_ptr<Slice>> getSlicesOrCreate(
        Timestamp timestamp, const std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>& createNewSlice) override;
    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>>
    getTriggerableWindowSlices(Timestamp globalWatermark) override;
    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> getAllNonTriggeredSlices() override;
    std::optional<std::shared_ptr<Slice>> getSliceBySliceEnd(SliceEnd sliceEnd) override;
    void garbageCollectSlicesAndWindows(Timestamp newGlobalWaterMark) override;
    void deleteState() override;
    void incrementNumberOfInputPipelines() override;
    uint64_t getWindowSize() const override;
    std::span<std::byte>
    allocateSpaceForSliceCache(uint64_t sliceCacheMemorySize, PipelineId pipelineId, AbstractBufferProvider& bufferProvider);

    /// Creates a SliceStoreRef that wraps this store. The store provides its own SliceCacheConfiguration;
    /// the caller only supplies the two operator-specific callbacks.
    std::unique_ptr<SliceStoreRef> createSliceStoreRef(
        DefaultTimeBasedSliceStoreRef::DataStructureExtractor extractor, DefaultTimeBasedSliceStoreRef::CreateSlicesFunction creator);

    /// Override the wall-clock source (default: steady_clock ms). Only for tests, to make the
    /// predictor-driven preemptive-create path deterministic.
    void setWallClockSourceForTesting(std::function<Timestamp()> clock);

private:
    SliceCacheConfiguration sliceCacheConfiguration;
    SlicePreallocationConfiguration slicePreallocationConfiguration;

    /// Slice-acquisition / preemptive-count strategies, selected once at construction (see the using
    /// aliases above). Set after the config members so the constructor can fall back to building them
    /// from slicePreallocationConfiguration.
    ObtainSliceFn obtainSliceFn;
    PreemptiveSliceCountFn preemptiveSliceCountFn;
    folly::Synchronized<std::unordered_map<PipelineId, std::unique_ptr<TupleBuffer>>> pipelineIdToSliceCacheStarts;

    /// We need to store the windows and slices in two separate maps. This is necessary as we need to access the slices during the join build phase,
    /// while we need to access windows during the triggering of windows.
    folly::Synchronized<std::map<WindowInfo, SlicesAndState>> windows;
    folly::Synchronized<std::map<SliceEnd, std::shared_ptr<Slice>>> slices;

    /// LIFO of slices retired by probe-side GC; build pops + Slice::reset()s instead of allocating
    /// a fresh slice. Drop-oldest (pop_front) when full. Empty + unused when recyclePoolSize == 0.
    folly::Synchronized<std::deque<std::shared_ptr<Slice>>> recyclePool;

    /// Predictor for the wall-clock preemptive-create horizon; null when preemptive_predictor == "off".
    /// Fed (event-time watermark, wall-clock) on each GC tick, read in the build-side create path —
    /// hence Synchronized (the predictor itself is not thread-safe).
    folly::Synchronized<std::unique_ptr<WatermarkPredictor>> preemptivePredictor;

    /// Wall-clock source for predictor observations + the create-horizon deadline. steady_clock ms by
    /// default; overridable in tests via setWallClockSourceForTesting.
    std::function<Timestamp()> wallClockNow;

    SliceAssigner sliceAssigner;

    /// We need to store the sequence number for the triggerable window infos. This is necessary, as we have to ensure that the sequence number is unique
    /// and increases for each window info.
    std::atomic<SequenceNumber::Underlying> sequenceNumber;

    /// If a window build operator appears in multiple pipelines, it may get terminated multiple times
    /// We need to track how many input pipelines have not terminated yet, to only release pending slices after the last termination
    std::atomic<uint64_t> numberOfActiveInputPipelines;
};

}
