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

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <StateReduction/CompressionAlgorithm.hpp>
#include <StateReduction/ReducibleSlice.hpp>
#include <StateReduction/SpillStore.hpp>
#include <StateReduction/StatePredictor.hpp>
#include <Time/Timestamp.hpp>
#include <Watermark/WatermarkPredictor.hpp>
#include <folly/Synchronized.h>
#include <StateReductionConfiguration.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

/// Owns state reduction for one windowed operator: the predictors, the compressor, the spill store, the
/// memory budget, and the bookkeeping for whatever is currently not resident.
///
/// The flow is one loop per build-side watermark advance. For each live slice it asks the
/// WatermarkPredictor when that slice will next be needed, hands the answer and the slice's size to the
/// StatePredictor, and moves the slice to whatever branch comes back. The memory budget then overrides
/// that answer at both ends: below the soft target nothing is reduced however cheap it looks, and above
/// the hard ceiling slices are reduced regardless of what the prediction says.
///
/// Slices that a pipeline is currently reading from or writing to are pinned and left alone. Pinning is
/// a count rather than a held lock because the build and probe paths run traced Nautilus code between
/// acquiring a slice and finishing with it, which a held mutex would have to span.
class StateReductionManager
{
public:
    /// @param windowSize the operator's window size, which is what turns a slice end into the time the
    ///        slice stops being needed. Without it there is no horizon to predict against.
    StateReductionManager(const StateReductionConfiguration& configuration, uint64_t operatorInstanceId, uint64_t windowSize);
    ~StateReductionManager();

    StateReductionManager(const StateReductionManager&) = delete;
    StateReductionManager(StateReductionManager&&) = delete;
    StateReductionManager& operator=(const StateReductionManager&) = delete;
    StateReductionManager& operator=(StateReductionManager&&) = delete;

    /// Measures what reduction costs on this machine. Called once, from the operator's start().
    void calibrate();

    /// The decision loop. Called on every build-side watermark advance, before the triggerable windows
    /// are collected.
    void onWatermarkAdvanced(Timestamp globalWatermark, std::span<const std::shared_ptr<Slice>> liveSlices);

    /// Brings a slice's state back if it is not resident, and marks it so the decision loop leaves it
    /// alone from then on. Safe to call on a slice that was never reduced, and on a slice type that
    /// does not support reduction.
    ///
    /// One-way on purpose. Callers pin a slice at the point they hand it to a probe, and from there its
    /// memory is kept alive by the trigger buffer regardless, so reducing it again would cost an encode
    /// and free nothing. The mark is dropped when the slice disappears from the store.
    ///
    /// @param bufferProvider the calling pipeline's provider, from its PipelineExecutionContext. Passed
    ///        per call rather than held, because a restore should allocate from the provider of whichever
    ///        thread is about to read the slice — that is the thread-local pool the caller's own buffers
    ///        come from, and it is not the same object for every caller or for the whole query lifetime.
    void pin(const std::shared_ptr<Slice>& slice, AbstractBufferProvider& bufferProvider);

    /// Drops any reduced state held for a slice that is being discarded.
    void forget(Timestamp sliceEnd);

    [[nodiscard]] bool isEnabled() const { return enabled; }

    /// Resident bytes across every slice the last decision loop saw, plus whatever the in-memory spill
    /// store is holding. This is the number the budget is compared against and the benchmarks plot.
    [[nodiscard]] uint64_t accountedBytes() const;

    /// Everything a run can be judged on afterwards, gathered per reduced-then-restored slice.
    ///
    /// Two different questions live here.
    ///
    /// *Was the decision worth making?* A slice is parked, then wanted again after `held`. Moving it out
    /// and back cost `roundTrip`. If it was wanted back sooner than that, the work bought nothing and the
    /// probe waited for a restore it need not have waited for. `totalHeld` and `totalRoundTrip` are kept
    /// raw alongside the miss count so a zero can be told apart from a broken comparison.
    ///
    /// *Was the WatermarkPredictor right?* Separately from what was decided, the predictor said the slice
    /// would not be needed for `predicted`; it was actually wanted after `held`. That error is the thing
    /// that distinguishes EWMA from Kalman, and it is recorded whatever the StatePredictor did with it —
    /// which makes a FORCED run the cleanest way to measure it, since there the decision does not depend
    /// on the prediction and so cannot bias the sample.
    ///
    /// Only slices that were actually reduced contribute. A slice never moved is never a data point.
    struct ReductionStats
    {
        uint64_t restores{0};

        uint64_t misses{0};
        std::chrono::nanoseconds totalMissed{0};
        std::chrono::nanoseconds worstMiss{0};
        std::chrono::nanoseconds totalHeld{0};
        std::chrono::nanoseconds totalRoundTrip{0};

        std::chrono::nanoseconds totalPredicted{0};
        std::chrono::nanoseconds totalAbsPredictionError{0};
        /// Signed, so systematic optimism and systematic pessimism do not cancel in the mean.
        std::chrono::nanoseconds signedPredictionError{0};
        /// Predicted far later than it was wanted: the predictor was too pessimistic.
        std::chrono::nanoseconds worstOverPrediction{0};
        /// Predicted far sooner than it was wanted: too optimistic, which is what causes misses.
        std::chrono::nanoseconds worstUnderPrediction{0};
    };

    [[nodiscard]] ReductionStats reductionStats() const;

    /// Appends this operator's summary row to the configured CSV and does nothing if no path was
    /// configured. Called when the operator stops; safe to call more than once, only the first writes.
    void writeStats();

private:
    /// What one non-resident slice is, and where.
    struct ReducedState
    {
        StateDecision decision{StateDecision::KeepInMemory};
        /// Only used by CompressInMemory; the spilling branches keep their bytes in the SpillStore.
        std::vector<std::byte> payload;
        uint64_t rawBytes{0};
        uint64_t storedBytes{0};
        /// When the state finished being parked, and what parking it cost. With the restore's own cost
        /// these are what decide whether the decision paid for itself.
        std::chrono::steady_clock::time_point parkedAt;
        std::chrono::nanoseconds reduceCost{0};
        /// What the WatermarkPredictor said, at the moment the decision was taken.
        std::chrono::nanoseconds predictedNeedIn{0};
        /// The slice this belongs to, so its liveness can be observed rather than inferred. Weak, because
        /// holding it would keep dead slices alive and defeat the store's garbage collection.
        std::weak_ptr<Slice> owner;
    };

    /// Carries the decision out rather than scheduling it: the compression and the store call run inline
    /// on the calling thread, so this blocks for however long the configured implementations take. That
    /// duration is a property of those implementations, not of this method. The cost is what the
    /// StatePredictor is meant to have already accounted for, which is why the work is done here:
    /// deferring it would only move the stall somewhere the cost model cannot see it.
    ///
    /// Returns the bytes this freed, so the caller can tell when it is back under budget without
    /// re-walking every slice.
    uint64_t applyDecision(
        const std::shared_ptr<Slice>& slice,
        ReducibleSlice& reducible,
        Timestamp sliceEnd,
        StateDecision decision,
        std::chrono::nanoseconds predictedNeedIn);

    /// The inverse, and blocking for the same reason, on the thread that is about to use the slice.
    void restore(ReducibleSlice& slice, Timestamp sliceEnd, AbstractBufferProvider& bufferProvider);
    /// Wall-clock milliseconds until the watermark is predicted to reach `lastNeededAt` -- the event time
    /// at which the slice stops being needed, not the one at which it stopped being written.
    [[nodiscard]] uint64_t predictedTimeUntilNeededInMs(Timestamp lastNeededAt) const;

    /// Drops bookkeeping for every tracked slice whose object has been destroyed.
    ///
    /// Liveness is read off the slice itself rather than off a snapshot of the store. A slice leaves the
    /// store's map before its last owner lets go: the trigger path holds shared_ptrs to the slices it is
    /// about to emit, so a slice can be absent from getAllSlices() and still be pinned and restored a
    /// moment later. Forgetting on absence dropped the reduced state out from under exactly that slice,
    /// leaving it flagged reduced with nothing to restore from.
    void forgetExpiredSlices();

    bool enabled;
    uint64_t memoryTargetBytes;
    uint64_t memoryCeilingBytes;
    uint32_t calibrationRepetitions;
    double learningRate{0.0};
    /// True between construction and calibrate() for a COST_MODEL operator. Until then statePredictor is
    /// a stand-in that reduces nothing.
    bool awaitingCalibration{false};

    std::unique_ptr<StatePredictor> statePredictor;
    std::unique_ptr<CompressionAlgorithm> compression;
    std::unique_ptr<SpillStore> spillStore;
    /// Guarded because observe() mutates it from the build thread while predictWallClock() is read from
    /// the same loop; WatermarkPredictor implementations are documented as not thread-safe.
    folly::Synchronized<std::unique_ptr<WatermarkPredictor>> watermarkPredictor;

    folly::Synchronized<std::unordered_map<uint64_t, ReducedState>> reducedStates;
    /// Slices that have been handed to a probe and are therefore off limits to the decision loop. Keyed
    /// by slice end, valued by the slice, so the entry can be dropped once the slice is gone.
    folly::Synchronized<std::unordered_map<uint64_t, std::weak_ptr<Slice>>> pinnedSliceEnds;

    std::atomic<uint64_t> residentBytesSeen{0};
    /// Written from probe threads on every restore, read by whoever is collecting numbers.
    folly::Synchronized<ReductionStats> stats;
    /// One counter per branch, so a run can be plotted by what the predictor actually chose rather than
    /// by what it was configured to choose -- the two differ as soon as a budget starts overriding it.
    std::array<std::atomic<uint64_t>, 4> reductionsByDecision{};
    std::atomic<uint64_t> bytesFreedTotal{0};

    uint64_t windowSize;
    std::string statsLogPath;
    /// Recorded so a plot can tell which predictor produced which errors.
    std::string watermarkPredictorName;
    uint64_t instanceId;
    std::atomic<bool> statsWritten{false};
    /// Atomic for the same reason as the handler's guard: the decision loop can be entered from more than
    /// one worker thread, and this is what keeps observe() from being fed the same watermark twice.
    std::atomic<Timestamp::Underlying> lastObservedWatermark{Timestamp::INVALID_VALUE};
};

}
