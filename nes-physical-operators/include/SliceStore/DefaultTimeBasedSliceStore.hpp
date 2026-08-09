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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include <SliceStoreConfiguration.hpp>

namespace NES
{


/// How many slices one winner should create for a claim: the missing slices of its buffer plus however
/// many the stream will demand while creation itself runs, so creation stays ahead of arrival.
/// watermarkRate is event-time ms per wall-clock ms; f is the fraction of one slide the stream advances
/// per created slice. Rate or cost of 0 (cold predictor, no measurements yet) means no speculation.
[[nodiscard]] inline uint64_t computeSliceGroupSize(
    const uint64_t neededSlices,
    const double watermarkRate,
    const uint64_t creationCostNanos,
    const uint64_t slideMs,
    const uint64_t maxGroupSize)
{
    if (neededSlices >= maxGroupSize)
    {
        return maxGroupSize;
    }
    if (watermarkRate <= 0.0 or creationCostNanos == 0 or slideMs == 0)
    {
        return neededSlices;
    }
    const double creationCostMs = static_cast<double>(creationCostNanos) / 1e6;
    const double advancePerSlice = watermarkRate * creationCostMs / static_cast<double>(slideMs);
    if (advancePerSlice >= 1.0)
    {
        /// The stream outruns creation: creating can never catch up, so create as much as allowed.
        return maxGroupSize;
    }
    const auto total = static_cast<uint64_t>(std::ceil(static_cast<double>(neededSlices) / (1.0 - advancePerSlice)));
    return std::clamp(total, neededSlices, maxGroupSize);
}

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
    DefaultTimeBasedSliceStore(
        uint64_t windowSize,
        uint64_t windowSlide,
        SliceCacheConfiguration sliceCacheConfiguration,
        SliceStoreConfiguration sliceStoreConfiguration);

    ~DefaultTimeBasedSliceStore() override;
    std::vector<std::shared_ptr<Slice>> getSlicesOrCreate(Timestamp timestamp, const SliceCreateFunction& createNewSlice) override;
    uint64_t
    claimOrDeferSliceRange(Timestamp minTs, Timestamp maxTs, double watermarkRate, const SliceCreateFunction& createNewSlice) override;
    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>>
    getTriggerableWindowSlices(Timestamp globalWatermark) override;
    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> getAllNonTriggeredSlices() override;
    std::optional<std::shared_ptr<Slice>> getSliceBySliceEnd(SliceEnd sliceEnd) override;
    std::vector<std::shared_ptr<Slice>> getAllSlices() override;
    void garbageCollectSlicesAndWindows(Timestamp newGlobalWaterMark) override;
    void deleteState() override;
    void incrementNumberOfInputPipelines() override;
    uint64_t getWindowSize() const override;
    [[nodiscard]] SliceStoreStatistics getStatistics() const override;
    std::span<std::byte>
    allocateSpaceForSliceCache(uint64_t sliceCacheMemorySize, PipelineId pipelineId, AbstractBufferProvider& bufferProvider);

    /// Creates a SliceStoreRef that wraps this store. The store provides its own SliceCacheConfiguration;
    /// the caller only supplies the two operator-specific callbacks.
    std::unique_ptr<SliceStoreRef> createSliceStoreRef(
        DefaultTimeBasedSliceStoreRef::DataStructureExtractor extractor, DefaultTimeBasedSliceStoreRef::CreateSlicesFunction creator);

private:
    SliceCacheConfiguration sliceCacheConfiguration;
    SliceStoreConfiguration sliceStoreConfiguration;
    folly::Synchronized<std::unordered_map<PipelineId, std::unique_ptr<TupleBuffer>>> pipelineIdToSliceCacheStarts;

    /// We need to store the windows and slices in two separate maps. This is necessary as we need to access the slices during the join build phase,
    /// while we need to access windows during the triggering of windows.
    folly::Synchronized<std::map<WindowInfo, SlicesAndState>> windows;
    folly::Synchronized<std::map<SliceEnd, std::shared_ptr<Slice>>> slices;
    SliceAssigner sliceAssigner;

    /// We need to store the sequence number for the triggerable window infos. This is necessary, as we have to ensure that the sequence number is unique
    /// and increases for each window info.
    std::atomic<SequenceNumber::Underlying> sequenceNumber;

    /// If a window build operator appears in multiple pipelines, it may get terminated multiple times
    /// We need to track how many input pipelines have not terminated yet, to only release pending slices after the last termination
    std::atomic<uint64_t> numberOfActiveInputPipelines;

    std::atomic<uint64_t> createdSlices{0};
    std::atomic<uint64_t> wastedSliceCreations{0};
    std::atomic<uint64_t> sliceCreationNanos{0};
    /// Rolling estimate of what one getSlicesOrCreate creation costs; feeds the group sizing.
    std::atomic<uint64_t> ewmaCreationNanos{0};

    /// Event-time ranges some thread is currently creating slices for. A claim lives only within one
    /// claimOrDeferSliceRange call, which is what makes the losers' retry condition guaranteed to clear.
    struct InFlightRange
    {
        uint64_t claimId{};
        Timestamp rangeStart{0};
        Timestamp rangeEnd{0};
        std::chrono::steady_clock::time_point expectedCompletion;
    };

    folly::Synchronized<std::vector<InFlightRange>> inFlightRanges;
    std::atomic<uint64_t> nextClaimId{0};

    /// Pushes the slice into the pool if recycling is on, we hold the only reference, and there is room.
    void poolRetiredSlice(std::shared_ptr<Slice> slice, bool pristine);
};

}
