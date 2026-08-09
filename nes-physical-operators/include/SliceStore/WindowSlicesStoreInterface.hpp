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

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <SliceStore/Slice.hpp>
#include <Time/Timestamp.hpp>
#include <folly/Synchronized.h>
#include <Util.hpp>

namespace NES
{

/// Stores the information (i.e., start and end timestamp) of a window
struct WindowInfo
{
    WindowInfo(const uint64_t windowStart, const uint64_t windowEnd) : windowStart(windowStart), windowEnd(windowEnd)
    {
        PRECONDITION(windowEnd >= windowStart, "Window end {} must be greater or equal to window start {}", windowEnd, windowStart);
    }

    bool operator<(const WindowInfo& other) const { return windowEnd < other.windowEnd; }

    Timestamp windowStart;
    Timestamp windowEnd;
};

/// This struct stores a slice and the window info
struct SlicesAndWindowInfo
{
    std::vector<std::shared_ptr<Slice>> windowSlices;
    WindowInfo windowInfo;
};

struct WindowInfoAndSequenceNumber
{
    WindowInfo windowInfo;
    SequenceNumber sequenceNumber;

    bool operator<(const WindowInfoAndSequenceNumber& other) const { return windowInfo < other.windowInfo; }
};

/// Lifetime counters of a slice store, logged once when the owning operator stops
struct SliceStoreStatistics
{
    uint64_t createdSlices{0};
    /// Slices that were fully built but discarded because another thread created the same slice concurrently
    uint64_t wastedSliceCreations{0};
    uint64_t sliceCreationNanos{0};
};

/// This is the interface for storing windows and slices in a window-based operator
/// It provides an interface to operate on slices and windows for a time-based window operator, e.g., join or aggregation
class WindowSlicesStoreInterface
{
public:
    virtual ~WindowSlicesStoreInterface() = default;

    [[nodiscard]] virtual SliceStoreStatistics getStatistics() const { return {}; }

    /// Called once for every slice whose identity dies (it is garbage collected), with its old slice end.
    /// Lets the owner drop per-slice-end bookkeeping (e.g. state reduction) before the slice is destroyed or reused.
    void setOnSliceRetired(std::function<void(SliceEnd)> callback) { onSliceRetired = std::move(callback); }

    /// Retrieves the slices that corresponds to the timestamp. If no slices exist for the timestamp, they are created by calling the method createNewSlice
    virtual std::vector<std::shared_ptr<Slice>> getSlicesOrCreate(Timestamp timestamp, const SliceCreateFunction& createNewSlice) = 0;

    /// Cooperative slice-group creation for the event-time range [minTs, maxTs] of one buffer.
    /// Returns 0 if the caller may proceed (all slices exist, or this caller won the claim and created
    /// them, possibly extended ahead of the stream by watermarkRate); otherwise another thread is
    /// creating an overlapping range and the returned value is the suggested retry delay in ms.
    virtual uint64_t claimOrDeferSliceRange(Timestamp, Timestamp, double, const SliceCreateFunction&) { return 0; }

    /// Retrieves all slices that can be triggered by the given global watermark
    /// This method returns all slices for each window that can be triggered. It returns the slices for all windows that have been filled and have a window end smaller than the global watermark
    /// Additionally, it returns a sequence number per window that is incremented for each window and thus, it can be used to set it in the emitted tuple buffer for the probe operator.
    virtual std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> getTriggerableWindowSlices(Timestamp globalWatermark)
        = 0;

    /// Retrieves the slice by its end timestamp. If no slice exists for the given slice end, the optional return value is nullopt
    virtual std::optional<std::shared_ptr<Slice>> getSliceBySliceEnd(SliceEnd sliceEnd) = 0;

    /// Retrieves every slice the store currently holds, triggered or not. Unlike the two methods above
    /// this says nothing about windows: it is the set of slices whose state is occupying memory right
    /// now, which is what state reduction reasons about.
    virtual std::vector<std::shared_ptr<Slice>> getAllSlices() = 0;

    /// Retrieves all current non-deleted slices that have not been triggered yet
    /// This method returns for each window all slices that have not been triggered yet, regardless of any watermark timestamp
    /// Additionally, it returns a sequence number per window that is incremented for each window and thus, it can be used to set it in the emitted tuple buffer for the probe operator.
    virtual std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> getAllNonTriggeredSlices() = 0;

    /// Garbage collect all slices and windows that are not valid anymore
    /// It is open for the implementation to delete the slices in this call or to mark them for deletion
    /// There is no guarantee that the slices are deleted after this call
    virtual void garbageCollectSlicesAndWindows(Timestamp newGlobalWaterMark) = 0;

    /// Deletes all slices, directly in this call
    virtual void deleteState() = 0;

    /// Increments the number of pipelines that contain a build(!) operator using this slice store, in order to track the expected number of terminations.
    /// This should be called each time an operator whose handler uses this store is set up.
    /// Note: This should not be inferred when the store is created during the lowering stage, as the same build operator may appear in multiple pipelines.
    virtual void incrementNumberOfInputPipelines() = 0;

    /// Returns the window size
    [[nodiscard]] virtual uint64_t getWindowSize() const = 0;

protected:
    WindowSlicesStoreInterface(const uint64_t slicePoolCapacity, const bool sliceRecyclingEnabled)
        : slicePoolCapacity(slicePoolCapacity), sliceRecyclingEnabled(sliceRecyclingEnabled)
    {
    }

    /// Retired slices waiting for reuse. Capped at the steady-state number of live slices; excess retirees
    /// are destroyed. Pooled hash-map slices keep the page high-water mark of their previous tenants.
    folly::Synchronized<std::vector<std::shared_ptr<Slice>>> slicePool;
    uint64_t slicePoolCapacity;
    bool sliceRecyclingEnabled;
    std::function<void(SliceEnd)> onSliceRetired;
};
}
