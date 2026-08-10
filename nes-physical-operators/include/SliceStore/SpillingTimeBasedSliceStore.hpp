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
#include <unordered_map>
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/MemoryPressureSensor.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <SliceStore/Spill/SpillStats.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Time/Timestamp.hpp>
#include <folly/Synchronized.h>

namespace NES
{

/// Decorator over a base WindowSlicesStoreInterface (typically DefaultTimeBasedSliceStore) that
/// adds disk-spill behaviour. The decorator is the ONLY component that talks to the SpillPolicy,
/// StorageBackend, MemoryPressureSensor, and SliceStateSerializerRegistry — everything else stays
/// in-memory-only.
///
/// Three-state handle tracking eliminates the double-spill race: each spilled slice's handle entry
/// is either SpillInFlight (spill future pending) or Spilled (handle present). Resident slices have
/// no entry. The decorator skips slices in SpillInFlight during GC ticks.
///
/// The set of candidate slices comes from `inner->getLiveSlices()` on every GC tick, NOT from
/// anything this decorator observes on its own methods. That is essential rather than incidental:
/// the JIT-compiled build path holds a SliceStoreRef bound to the concrete inner store and calls
/// getSlicesOrCreate on it directly, so a decorator that tracked its own callers would see almost
/// nothing and would never spill.
///
/// Restore on probe access is BLOCKING in v1. A future change can hand the restore future to the
/// pipeline scheduler so the worker can run other tasks while the I/O is in flight.
class SpillingTimeBasedSliceStore final : public WindowSlicesStoreInterface
{
public:
    SpillingTimeBasedSliceStore(
        std::unique_ptr<WindowSlicesStoreInterface> inner,
        std::unique_ptr<SpillPolicy> policy,
        std::shared_ptr<StorageBackend> backend,
        std::unique_ptr<MemoryPressureSensor> sensor,
        AbstractBufferProvider& buffers,
        SliceStateSerializer& serializer);

    ~SpillingTimeBasedSliceStore() override;

    std::vector<std::shared_ptr<Slice>> getSlicesOrCreate(
        Timestamp timestamp, const std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>& createNewSlice) override;

    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>>
    getTriggerableWindowSlices(Timestamp globalWatermark) override;

    std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>> getAllNonTriggeredSlices() override;

    [[nodiscard]] std::vector<std::shared_ptr<Slice>> getLiveSlices() const override;

    std::optional<std::shared_ptr<Slice>> getSliceBySliceEnd(SliceEnd sliceEnd) override;

    void garbageCollectSlicesAndWindows(Timestamp newGlobalWaterMark) override;

    void deleteState() override;

    void incrementNumberOfInputPipelines() override;

    [[nodiscard]] uint64_t getWindowSize() const override;

    /// Test-only: returns whether a handle exists for the given slice end.
    [[nodiscard]] bool isSliceSpilled(SliceEnd sliceEnd) const noexcept;

    /// Test-only: returns the number of currently-spilled slices.
    [[nodiscard]] std::size_t numSpilledSlices() const noexcept;

    /// Live counters for this store. Read by SpillStatsRegistry's sampler and by tests.
    [[nodiscard]] const SpillStatistics& statistics() const noexcept { return stats; }

    /// Override the wall-clock source (default: steady_clock ms). Only for tests, to make the
    /// predictor-driven preemptive-create path deterministic.
    void setWallClockSourceForTesting(std::function<Timestamp()> clock);

private:
    enum class HandleState : uint8_t
    {
        SpillInFlight,
        Spilled,
    };

    struct HandleEntry
    {
        HandleState state{HandleState::SpillInFlight};
        SpilledSliceHandle handle{};
    };

    /// Restore a spilled slice synchronously. Throws CannotDeserialize if the backend restore
    /// fails — a spilled slice we cannot read back is lost state.
    void restoreSliceSynchronous(Slice& slice, const SpilledSliceHandle& handle);

    /// Spill a slice synchronously. Returns the handle on success, or nullopt when the serializer
    /// declined or failed — in which case the slice's pages are still intact and it stays resident.
    /// Never throws: a slice that cannot be spilled is not a query error, only lost headroom.
    std::optional<SpilledSliceHandle> spillSliceSynchronous(Slice& slice);

    std::unique_ptr<WindowSlicesStoreInterface> inner;
    std::unique_ptr<SpillPolicy> spillPolicy;
    std::shared_ptr<StorageBackend> backend;
    std::unique_ptr<MemoryPressureSensor> sensor;
    AbstractBufferProvider& buffers;
    /// A store only ever holds one Slice subclass, so the serializer is resolved once at construction
    /// (from the slice type known at lowering) instead of a registry lookup per spill/restore/GC.
    SliceStateSerializer& serializer;

    /// Spill state keyed by slice end: each entry is either SpillInFlight or Spilled. Resident
    /// (in-memory) slices have no entry here.
    folly::Synchronized<std::unordered_map<Timestamp::Underlying, HandleEntry>> spillHandlesBySliceEnd;

    /// Wall-clock source for predictor observations + the create-horizon deadline. steady_clock ms by
    /// default; overridable in tests via setWallClockSourceForTesting.
    std::function<Timestamp()> wallClockNow;

    /// Spill/restore telemetry. Registered with SpillStatsRegistry for the lifetime of this store, so
    /// it must outlive nothing else here — it is declared before the id that unregisters it.
    SpillStatistics stats;
    uint64_t statsRegistrationId{0};
};

}
