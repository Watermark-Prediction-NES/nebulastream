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

#include <StateReduction/StateReductionManager.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <system_error>
#include <utility>
#include <vector>
#include <unistd.h>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <StateReduction/CompressionAlgorithm.hpp>
#include <StateReduction/CostModelStatePredictor.hpp>
#include <StateReduction/ForcedStatePredictor.hpp>
#include <StateReduction/ReducibleSlice.hpp>
#include <StateReduction/SpillStore.hpp>
#include <StateReduction/StatePredictor.hpp>
#include <StateReduction/StateReductionCalibrator.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/Logger.hpp>
#include <Watermark/WatermarkPredictor.hpp>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include <ErrorHandling.hpp>
#include <StateReductionConfiguration.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

namespace
{

/// Slices that do not implement ReducibleSlice are simply never chosen. That is a no-op rather than an
/// error, so enabling state reduction on a query mixing reducible and non-reducible operators stays
/// safe.
ReducibleSlice* asReducible(Slice& slice)
{
    return dynamic_cast<ReducibleSlice*>(&slice);
}

Timestamp wallClockNow()
{
    const auto sinceEpoch = std::chrono::steady_clock::now().time_since_epoch();
    return Timestamp{static_cast<Timestamp::Underlying>(std::chrono::duration_cast<std::chrono::milliseconds>(sinceEpoch).count())};
}

bool spillsToStore(const StateDecision decision)
{
    return decision == StateDecision::Spill || decision == StateDecision::CompressAndSpill;
}

}

StateReductionManager::StateReductionManager(
    const StateReductionConfiguration& configuration, const uint64_t operatorInstanceId, const uint64_t windowSize)
    : enabled(configuration.enabled.getValue())
    , memoryTargetBytes(configuration.memoryTargetBytes.getValue())
    , memoryCeilingBytes(configuration.memoryCeilingBytes.getValue())
    , calibrationRepetitions(static_cast<uint32_t>(configuration.calibrationRepetitions.getValue()))
    , windowSize(windowSize)
    , statsLogPath(configuration.statsLogPath.getValue())
    , watermarkPredictorName(magic_enum::enum_name(configuration.watermarkPredictor.getValue()))
    , instanceId(operatorInstanceId)
{
    if (not enabled)
    {
        return;
    }

    PRECONDITION(
        memoryCeilingBytes == 0 || memoryTargetBytes == 0 || memoryCeilingBytes >= memoryTargetBytes,
        "operator_memory_ceiling_bytes ({}) must not be below operator_memory_target_bytes ({})",
        memoryCeilingBytes,
        memoryTargetBytes);

    compression
        = CompressionAlgorithm::create(configuration.compression.getValue(), static_cast<int>(configuration.compressionLevel.getValue()));

    /// Each operator gets its own subdirectory, keyed by process as well as by operator. Slice ends are
    /// only unique within one operator, so the subdirectory is what keeps two operators from reading each
    /// other's state — and the pid is what keeps two *processes* sharing a spill root from doing the
    /// same. The instance id is only a process-local counter, so without the pid two workers started
    /// against the same root both claim "operator-0", and LocalFileSpillStore clears its directory on
    /// construction: the second one deletes the first one's state while it is still running.
    const auto directory = std::filesystem::path{configuration.spillDirectory.getValue()}
        / fmt::format("worker-{}-operator-{}", ::getpid(), operatorInstanceId);
    spillStore = SpillStore::create(configuration.spillStore.getValue(), directory);

    watermarkPredictor = WatermarkPredictor::create(configuration.watermarkPredictor.getValue());

    learningRate = configuration.learningRate.getValue();
    switch (configuration.predictor.getValue())
    {
        case StatePredictorType::FORCED:
            statePredictor = std::make_unique<ForcedStatePredictor>(configuration.forcedDecision.getValue());
            break;
        case StatePredictorType::COST_MODEL:
            /// Deliberately not a cost model yet. An uncalibrated model believes every stage is free and
            /// would therefore reduce everything, so until calibrate() has run the safe answer to every
            /// question is "leave it alone".
            statePredictor = std::make_unique<ForcedStatePredictor>(StateDecision::KeepInMemory);
            awaitingCalibration = true;
            break;
    }
}

StateReductionManager::~StateReductionManager() = default;

void StateReductionManager::calibrate()
{
    /// Only the cost model has anything to calibrate. A forced predictor ignores the numbers, so paying
    /// for the measurement would only slow down the baseline every benchmark is compared against.
    if (not enabled || not awaitingCalibration || compression == nullptr || spillStore == nullptr)
    {
        return;
    }

    const auto model
        = calibrateStateReduction(*compression, *spillStore, CalibrationConfig{.repetitions = std::max(calibrationRepetitions, 1U)});
    statePredictor = std::make_unique<CostModelStatePredictor>(model, learningRate);
    awaitingCalibration = false;
}

uint64_t StateReductionManager::predictedTimeUntilNeededInMs(const Timestamp lastNeededAt) const
{
    const auto predicted = watermarkPredictor.withRLock(
        [lastNeededAt](const auto& predictor)
        { return predictor ? predictor->predictWallClock(lastNeededAt) : Timestamp{Timestamp::INVALID_VALUE}; });

    /// No predictor, or not enough observations yet. Reporting zero means "needed now", which only
    /// KeepInMemory satisfies — so an unwarmed system never reduces state it cannot get back.
    if (predicted == Timestamp{Timestamp::INVALID_VALUE})
    {
        return 0;
    }

    const auto now = wallClockNow();
    if (predicted <= now)
    {
        return 0;
    }
    return predicted.getRawValue() - now.getRawValue();
}

void StateReductionManager::onWatermarkAdvanced(const Timestamp globalWatermark, const std::span<const std::shared_ptr<Slice>> liveSlices)
{
    /// No provider needed here: every branch of the decision tree only ever releases buffers. Allocation
    /// happens exclusively on the restore path, which runs from pin() with the caller's own provider.
    if (not enabled)
    {
        return;
    }

    if (lastObservedWatermark.exchange(globalWatermark.getRawValue(), std::memory_order_relaxed) != globalWatermark.getRawValue())
    {
        watermarkPredictor.withWLock(
            [globalWatermark](auto& predictor)
            {
                if (predictor)
                {
                    predictor->observe(globalWatermark, wallClockNow());
                }
            });
    }

    /// One pass to size the operator, because the budget is a property of the whole operator rather
    /// than of any single slice.
    struct Candidate
    {
        std::shared_ptr<Slice> owner;
        ReducibleSlice* slice;
        Timestamp sliceEnd;
        /// The event time at which the last window containing this slice ends, which is when the slice
        /// stops being needed. SliceAssigner derives the same value as sliceStart + windowSize.
        Timestamp lastNeededAt;
        uint64_t residentBytes;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(liveSlices.size());

    uint64_t totalResident = 0;
    for (const auto& slicePtr : liveSlices)
    {
        if (not slicePtr)
        {
            continue;
        }
        const auto sliceEnd = slicePtr->getSliceEnd();

        auto* const reducible = asReducible(*slicePtr);
        if (reducible == nullptr)
        {
            continue;
        }

        /// A slice that can still receive records must not be touched at all -- not reduced, and not even
        /// measured. The build path writes through raw TupleBuffer pointers cached per worker thread and
        /// takes no lock, by design, so anything here that walks a slice's buffer tree races with a build
        /// thread appending pages and variable-sized children: the child count is read, the vector grows,
        /// and the load of the next index is either out of range or a torn read. stateMutex() does not
        /// help, because the build side never acquires it.
        ///
        /// The watermark is what rules that out. No future record can fall into a slice that ends at or
        /// before it, and it cannot advance past a buffer that is still in flight, so a slice past the
        /// watermark is settled. Slices still needed by open *windows* are a different matter and are
        /// exactly what this is here to reduce -- a slice stops being written long before the last window
        /// containing it is triggered.
        ///
        /// The cost is that the budget only sees settled slices. Counting the ones still filling would
        /// mean either walking them unsynchronised, which is the crash above, or synchronising the build
        /// path, which is the throughput the whole design exists to protect.
        if (sliceEnd > globalWatermark)
        {
            continue;
        }

        const std::shared_lock readGuard{reducible->stateMutex()};
        const auto bytes = reducible->residentBytes();
        totalResident += bytes;

        if (reducible->isReduced())
        {
            continue;
        }

        if (pinnedSliceEnds.withRLock([sliceEnd](const auto& pinned) { return pinned.contains(sliceEnd.getRawValue()); }))
        {
            /// Already handed to a probe.
            continue;
        }

        candidates.push_back(
            {.owner = slicePtr,
             .slice = reducible,
             .sliceEnd = sliceEnd,
             .lastNeededAt = Timestamp{slicePtr->getSliceStart().getRawValue() + windowSize},
             .residentBytes = bytes});
    }
    residentBytesSeen.store(totalResident, std::memory_order_relaxed);
    forgetExpiredSlices();

    if (memoryTargetBytes != 0 && totalResident <= memoryTargetBytes)
    {
        /// Under the soft budget there is nothing to buy by reducing, and every reduction is a restore
        /// waiting to happen on the probe path.
        return;
    }

    /// Largest first. Over the ceiling this frees the most memory per reduction; under it the order is
    /// immaterial because every candidate is considered anyway.
    std::ranges::sort(candidates, [](const Candidate& lhs, const Candidate& rhs) { return lhs.residentBytes > rhs.residentBytes; });

    uint64_t projectedResident = totalResident;
    for (const auto& candidate : candidates)
    {
        const auto overCeiling = memoryCeilingBytes != 0 && projectedResident > memoryCeilingBytes;

        /// The horizon is until the slice stops being *needed*, not until it stopped being written. Asking
        /// about sliceEnd is always answered "already passed", because this loop only ever sees slices the
        /// watermark has overtaken -- which silently pinned every deadline to zero and so let the cost
        /// model choose nothing but KeepInMemory. For a tumbling window the two coincide and the horizon
        /// really is zero; a sliding window is where there is anything to exploit.
        const auto predictedNeedInMs = predictedTimeUntilNeededInMs(candidate.lastNeededAt);
        auto decision = statePredictor->predict(candidate.residentBytes, predictedNeedInMs);
        if (overCeiling && decision == StateDecision::KeepInMemory)
        {
            /// The ceiling is not advisory. Above it we take the branch that frees everything even
            /// though the prediction says we will pay for it on the probe, because the alternative is
            /// breaching a budget the operator was told to live within. With compression configured to
            /// NONE this is Spill in all but name, so there is no second case to handle.
            decision = StateDecision::CompressAndSpill;
        }
        if (decision == StateDecision::KeepInMemory)
        {
            continue;
        }

        const auto freed
            = applyDecision(candidate.owner, *candidate.slice, candidate.sliceEnd, decision, std::chrono::milliseconds{predictedNeedInMs});
        projectedResident -= std::min(freed, projectedResident);

        if (memoryTargetBytes != 0 && projectedResident <= memoryTargetBytes)
        {
            break;
        }
    }
}

uint64_t StateReductionManager::applyDecision(
    const std::shared_ptr<Slice>& owner,
    ReducibleSlice& slice,
    const Timestamp sliceEnd,
    const StateDecision decision,
    const std::chrono::nanoseconds predictedNeedIn)
{
    const std::unique_lock writeGuard{slice.stateMutex()};
    if (slice.isReduced())
    {
        return 0;
    }
    /// Re-check under the lock: pin() takes the same mutex, so a pipeline that grabbed this slice
    /// between the survey above and here has already been serialised against us.
    if (pinnedSliceEnds.withRLock([sliceEnd](const auto& pinned) { return pinned.contains(sliceEnd.getRawValue()); }))
    {
        return 0;
    }

    const auto rawBytesBefore = slice.residentBytes();
    const auto start = std::chrono::steady_clock::now();

    std::vector<std::byte> raw;
    slice.serializeState(raw);

    ReducedState entry;
    entry.decision = decision;
    entry.rawBytes = raw.size();

    switch (decision)
    {
        case StateDecision::CompressInMemory:
            entry.payload = compression->compress(raw);
            entry.storedBytes = entry.payload.size();
            break;
        case StateDecision::Spill:
            spillStore->put(sliceEnd.getRawValue(), raw);
            entry.storedBytes = raw.size();
            break;
        case StateDecision::CompressAndSpill: {
            const auto compressed = compression->compress(raw);
            spillStore->put(sliceEnd.getRawValue(), compressed);
            entry.storedBytes = compressed.size();
            break;
        }
        case StateDecision::KeepInMemory:
            std::unreachable();
    }

    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto storedBytes = entry.storedBytes;
    entry.reduceCost = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
    entry.parkedAt = std::chrono::steady_clock::now();
    entry.predictedNeedIn = predictedNeedIn;
    entry.owner = owner;
    reducedStates.withWLock([&](auto& states) { states[sliceEnd.getRawValue()] = std::move(entry); });
    reductionsByDecision.at(static_cast<size_t>(decision)).fetch_add(1, std::memory_order_relaxed);

    /// Measured rather than assumed: what the compressor actually achieved on this slice's data is the
    /// only thing that can correct the ratio the start-up calibration guessed from synthetic bytes.
    /// CompressInMemory still holds its compressed copy on the heap, so that copy is not freed memory.
    const auto residentAfter = slice.residentBytes();
    const auto inMemoryCost = decision == StateDecision::CompressInMemory ? storedBytes : 0;
    const auto heldAfter = residentAfter + inMemoryCost;
    const auto freed = rawBytesBefore > heldAfter ? rawBytesBefore - heldAfter : 0;
    const auto reducedPercentage = rawBytesBefore > 0 ? static_cast<double>(freed) / static_cast<double>(rawBytesBefore) : 0.0;
    statePredictor->update(decision, reducedPercentage, rawBytesBefore, std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed));
    bytesFreedTotal.fetch_add(freed, std::memory_order_relaxed);
    return freed;
}

void StateReductionManager::restore(ReducibleSlice& slice, const Timestamp sliceEnd, AbstractBufferProvider& bufferProvider)
{
    ReducedState entry;
    const auto found = reducedStates.withWLock(
        [&](auto& states)
        {
            const auto it = states.find(sliceEnd.getRawValue());
            if (it == states.end())
            {
                return false;
            }
            entry = std::move(it->second);
            states.erase(it);
            return true;
        });
    INVARIANT(found, "Slice ending at {} reports itself reduced but no reduced state is held for it", sliceEnd);

    const auto start = std::chrono::steady_clock::now();

    std::vector<std::byte> raw;
    switch (entry.decision)
    {
        case StateDecision::CompressInMemory:
            raw = compression->decompress(entry.payload, entry.rawBytes);
            break;
        case StateDecision::Spill:
            raw = spillStore->get(sliceEnd.getRawValue());
            break;
        case StateDecision::CompressAndSpill:
            raw = compression->decompress(spillStore->get(sliceEnd.getRawValue()), entry.rawBytes);
            break;
        case StateDecision::KeepInMemory:
            std::unreachable();
    }

    slice.deserializeState(raw, bufferProvider);

    if (spillsToStore(entry.decision))
    {
        spillStore->erase(sliceEnd.getRawValue());
    }

    const auto elapsed = std::chrono::steady_clock::now() - start;
    statePredictor->updateAfterRestore(entry.decision, entry.storedBytes, std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed));

    /// Was moving this state worth it? The slice sat parked for `heldFor`; moving it out and back cost
    /// `roundTrip`. If it was wanted again sooner than the round trip took, the work bought nothing and
    /// the probe waited for a restore it need not have waited for -- the deadline we were given was wrong.
    ///
    /// Measured rather than taken from the cost model on purpose: the model's estimate is the thing
    /// update()/updateAfterRestore() are already correcting, so scoring the decision against it would
    /// only say how self-consistent the model is. These two durations are what actually happened.
    ///
    /// `start`, not now(): the state was needed the moment this restore was asked for. Measuring to the
    /// end would fold the restore into the idle period and compare it against itself, which quietly
    /// weakens the test to `idle < reduceCost` and reports far fewer misses than there were.
    const auto heldFor = std::chrono::duration_cast<std::chrono::nanoseconds>(start - entry.parkedAt);
    const auto roundTrip = entry.reduceCost + std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
    /// The second question, independent of what was decided: how far out was the WatermarkPredictor?
    /// `heldFor` is the ground truth it was predicting, so this is its error on this slice.
    const auto predictionError = heldFor - entry.predictedNeedIn;

    stats.withWLock(
        [heldFor, roundTrip, predictionError, predicted = entry.predictedNeedIn](auto& current)
        {
            ++current.restores;
            current.totalHeld += heldFor;
            current.totalRoundTrip += roundTrip;
            if (heldFor < roundTrip)
            {
                const auto missedBy = roundTrip - heldFor;
                ++current.misses;
                current.totalMissed += missedBy;
                current.worstMiss = std::max(current.worstMiss, missedBy);
            }

            current.totalPredicted += predicted;
            current.totalAbsPredictionError += std::chrono::nanoseconds{std::abs(predictionError.count())};
            current.signedPredictionError += predictionError;
            if (predictionError < std::chrono::nanoseconds{0})
            {
                current.worstOverPrediction = std::max(current.worstOverPrediction, -predictionError);
            }
            else
            {
                current.worstUnderPrediction = std::max(current.worstUnderPrediction, predictionError);
            }
        });
}

void StateReductionManager::pin(const std::shared_ptr<Slice>& slice, AbstractBufferProvider& bufferProvider)
{
    if (not enabled || not slice)
    {
        return;
    }
    auto* const reducible = asReducible(*slice);
    if (reducible == nullptr)
    {
        return;
    }

    const std::unique_lock writeGuard{reducible->stateMutex()};
    /// Taken under the same mutex applyDecision uses, so a reduction cannot start between the check and
    /// the mark.
    pinnedSliceEnds.withWLock([&](auto& pinned) { pinned.insert_or_assign(slice->getSliceEnd().getRawValue(), slice); });
    if (reducible->isReduced())
    {
        restore(*reducible, slice->getSliceEnd(), bufferProvider);
    }
}

void StateReductionManager::forgetExpiredSlices()
{
    /// Garbage collection happens inside the slice store, which knows nothing about this manager, so the
    /// decision loop cleans up after it. Without this a long-running query would accumulate spill files
    /// and map entries for slices that are gone.
    ///
    /// The test is the slice's own lifetime, not its presence in the store: a slice is removed from the
    /// store's map while the trigger path still holds a shared_ptr to it, and that is precisely the slice
    /// that is about to be pinned and restored.
    std::vector<uint64_t> stale;
    reducedStates.withRLock(
        [&](const auto& states)
        {
            for (const auto& [sliceEnd, state] : states)
            {
                if (state.owner.expired())
                {
                    stale.push_back(sliceEnd);
                }
            }
        });
    pinnedSliceEnds.withRLock(
        [&](const auto& pinned)
        {
            for (const auto& [sliceEnd, owner] : pinned)
            {
                if (owner.expired())
                {
                    stale.push_back(sliceEnd);
                }
            }
        });

    for (const auto sliceEnd : stale)
    {
        forget(Timestamp{sliceEnd});
    }
}

void StateReductionManager::forget(const Timestamp sliceEnd)
{
    if (not enabled)
    {
        return;
    }
    reducedStates.withWLock([&](auto& states) { states.erase(sliceEnd.getRawValue()); });
    pinnedSliceEnds.withWLock([&](auto& pinned) { pinned.erase(sliceEnd.getRawValue()); });
    if (spillStore)
    {
        spillStore->erase(sliceEnd.getRawValue());
    }
}

StateReductionManager::ReductionStats StateReductionManager::reductionStats() const
{
    return stats.copy();
}

void StateReductionManager::writeStats()
{
    if (statsLogPath.empty() || statsWritten.exchange(true, std::memory_order_relaxed))
    {
        return;
    }

    const auto collected = reductionStats();
    const std::filesystem::path path{statsLogPath};
    std::error_code errorCode;
    if (path.has_parent_path())
    {
        std::filesystem::create_directories(path.parent_path(), errorCode);
    }

    /// Header only for a file nobody has written to yet, so a whole sweep can append to one path.
    const auto existingSize = std::filesystem::file_size(path, errorCode);
    const bool needsHeader = errorCode || existingSize == 0;

    std::ofstream file{path, std::ios::app};
    if (not file.is_open())
    {
        NES_WARNING("Could not open {} to write state-reduction stats", statsLogPath);
        return;
    }
    if (needsHeader)
    {
        file << "operator_instance,watermark_predictor,kept_in_memory,compressed_in_memory,spilled,compressed_and_spilled,"
                "bytes_freed_total,resident_bytes_last,restores,total_held_ns,total_round_trip_ns,deadline_misses,"
                "total_missed_ns,worst_miss_ns,total_predicted_ns,total_abs_prediction_error_ns,"
                "signed_prediction_error_ns,worst_over_prediction_ns,worst_under_prediction_ns\n";
    }

    /// One row per operator, written once when it stops. Short enough that a single append from several
    /// operators -- or several workers sharing a path -- does not interleave in practice.
    file << instanceId << ',' << watermarkPredictorName << ','
         << reductionsByDecision.at(static_cast<size_t>(StateDecision::KeepInMemory)).load() << ','
         << reductionsByDecision.at(static_cast<size_t>(StateDecision::CompressInMemory)).load() << ','
         << reductionsByDecision.at(static_cast<size_t>(StateDecision::Spill)).load() << ','
         << reductionsByDecision.at(static_cast<size_t>(StateDecision::CompressAndSpill)).load() << ',' << bytesFreedTotal.load() << ','
         << residentBytesSeen.load() << ',' << collected.restores << ',' << collected.totalHeld.count() << ','
         << collected.totalRoundTrip.count() << ',' << collected.misses << ',' << collected.totalMissed.count() << ','
         << collected.worstMiss.count() << ',' << collected.totalPredicted.count() << ',' << collected.totalAbsPredictionError.count()
         << ',' << collected.signedPredictionError.count() << ',' << collected.worstOverPrediction.count() << ','
         << collected.worstUnderPrediction.count() << '\n';
}

uint64_t StateReductionManager::accountedBytes() const
{
    const auto resident = residentBytesSeen.load(std::memory_order_relaxed);
    /// An in-memory spill store still holds the operator's memory, just compacted, so the budget has to
    /// see it. A local-file store holds disk, which the budget is not about.
    return resident + (spillStore ? spillStore->storedBytes() : 0);
}

}
