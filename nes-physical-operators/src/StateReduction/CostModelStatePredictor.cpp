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

#include <StateReduction/CostModelStatePredictor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <ranges>
#include <utility>
#include <StateReduction/StateReductionCostModel.hpp>
#include <ErrorHandling.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

namespace
{
constexpr std::array AllDecisions
    = {StateDecision::KeepInMemory, StateDecision::CompressInMemory, StateDecision::Spill, StateDecision::CompressAndSpill};

/// Compression can only ever be estimated, never known in advance, so the ratio is clamped away from the
/// degenerate ends: 0 would make compressed state look free, and above 1 would mean compressing grew it,
/// which the rest of the model has no way to act on.
constexpr double MinCompressionRatio = 0.01;
constexpr double MaxCompressionRatio = 1.0;

/// Bounds how far one observation may move the model. A timing that caught a page fault or a scheduler
/// preemption is not evidence that the machine got ten times slower, and without a bound a single such
/// sample can push the model far enough that it stops reducing anything — after which it never gets
/// another observation to recover from.
constexpr double MinCorrection = 0.1;
constexpr double MaxCorrection = 10.0;

/// Resident bytes left after applying `decision` to `bytes` of state, given an expected compression
/// ratio in (0, 1].
uint64_t residentBytesAfter(const StateDecision decision, const uint64_t bytes, const double compressionRatio)
{
    switch (decision)
    {
        case StateDecision::KeepInMemory:
            return bytes;
        case StateDecision::CompressInMemory:
            return static_cast<uint64_t>(static_cast<double>(bytes) * compressionRatio);
        case StateDecision::Spill:
        case StateDecision::CompressAndSpill:
            /// The pages are handed back to the buffer provider, so nothing of this slice stays resident.
            return 0;
    }
    std::unreachable();
}

/// Ratio of what a stage actually took to what the model expected, bounded. Returns nullopt when the
/// model predicted nothing, in which case there is no factor to correct by.
std::optional<double> correctionFactor(const double predictedNanos, const std::chrono::nanoseconds elapsed)
{
    const auto observedNanos = static_cast<double>(elapsed.count());
    if (predictedNanos <= 0.0 || observedNanos <= 0.0)
    {
        return std::nullopt;
    }
    return std::clamp(observedNanos / predictedNanos, MinCorrection, MaxCorrection);
}
}

CostModelStatePredictor::CostModelStatePredictor(StateReductionCostModel initialModel, const double learningRate)
    : model(std::move(initialModel)), learningRate(std::clamp(learningRate, 0.0, 1.0))
{
    PRECONDITION(
        model.compressionRatio > 0.0 && model.compressionRatio <= 1.0, "Compression ratio {} must be in (0, 1]", model.compressionRatio);
}

StateDecision CostModelStatePredictor::predict(const uint64_t bytes, const uint64_t maxRequiredTimeInMilliSeconds) const
{
    StateReductionCostModel snapshot;
    {
        const std::scoped_lock lock{modelMutex};
        snapshot = model;
    }

    const auto budgetNanos = static_cast<double>(maxRequiredTimeInMilliSeconds) * 1'000'000.0;

    /// Free the most memory; among options that free the same amount, pay the least for it. That second
    /// clause is what decides between Spill and CompressAndSpill, which both free everything, and it is
    /// what the pair's lexicographic ordering expresses.
    /// KeepInMemory costs nothing, so it always survives the filter and the range is never empty.
    /// Not const: filter_view caches its begin(), so begin() is non-const and a const filter_view does not
    /// model range.
    auto affordable
        = AllDecisions | std::views::filter([&](const auto decision) { return snapshot.roundTripNanos(decision, bytes) <= budgetNanos; });
    return *std::ranges::min_element(
        affordable,
        {},
        [&](const auto decision)
        { return std::pair{residentBytesAfter(decision, bytes, snapshot.compressionRatio), snapshot.roundTripNanos(decision, bytes)}; });
}

void CostModelStatePredictor::update(
    const StateDecision decision, const double reducedPercentage, const uint64_t bytes, const std::chrono::nanoseconds elapsed)
{
    if (decision == StateDecision::KeepInMemory || bytes == 0)
    {
        return;
    }

    const std::scoped_lock lock{modelMutex};

    /// A reduction is the only place we learn what the data actually compresses to. Spill stores the
    /// state verbatim, so its reducedPercentage says nothing about compressibility and is ignored here.
    if (decision == StateDecision::CompressInMemory || decision == StateDecision::CompressAndSpill)
    {
        const double observedRatio = std::clamp(1.0 - reducedPercentage, MinCompressionRatio, MaxCompressionRatio);
        model.compressionRatio += learningRate * (observedRatio - model.compressionRatio);
    }

    /// `elapsed` covers every stage the decision involves, and one timing cannot be split between them.
    /// So every stage the decision touched is scaled by the same factor the total was off by, which
    /// leaves their relative contributions as calibration measured them.
    const auto correction = correctionFactor(model.reduceNanos(decision, bytes), elapsed);
    if (not correction.has_value())
    {
        return;
    }

    switch (decision)
    {
        case StateDecision::CompressInMemory:
            model.compress.applyCorrection(*correction, learningRate);
            break;
        case StateDecision::Spill:
            model.spillWrite.applyCorrection(*correction, learningRate);
            break;
        case StateDecision::CompressAndSpill:
            model.compress.applyCorrection(*correction, learningRate);
            model.spillWrite.applyCorrection(*correction, learningRate);
            break;
        case StateDecision::KeepInMemory:
            std::unreachable();
    }
}

void CostModelStatePredictor::updateAfterRestore(
    const StateDecision decision, const uint64_t reducedBytes, const std::chrono::nanoseconds elapsed)
{
    if (decision == StateDecision::KeepInMemory || reducedBytes == 0)
    {
        return;
    }

    const std::scoped_lock lock{modelMutex};

    const auto correction = correctionFactor(model.restoreNanos(decision, reducedBytes), elapsed);
    if (not correction.has_value())
    {
        return;
    }

    switch (decision)
    {
        case StateDecision::CompressInMemory:
            model.decompress.applyCorrection(*correction, learningRate);
            break;
        case StateDecision::Spill:
            model.spillRead.applyCorrection(*correction, learningRate);
            break;
        case StateDecision::CompressAndSpill:
            model.spillRead.applyCorrection(*correction, learningRate);
            model.decompress.applyCorrection(*correction, learningRate);
            break;
        case StateDecision::KeepInMemory:
            std::unreachable();
    }
}

StateReductionCostModel CostModelStatePredictor::currentModel() const
{
    const std::scoped_lock lock{modelMutex};
    return model;
}

}
