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

#include <StateReduction/StateReductionCalibrator.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>
#include <StateReduction/CompressionAlgorithm.hpp>
#include <StateReduction/SpillStore.hpp>
#include <StateReduction/StateReductionCostModel.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

namespace
{

/// Slice ends calibration writes under. Counting down from the top keeps them clear of any real slice
/// end, which is a wall-clock or event-time value; each one is erased again before calibration returns.
constexpr uint64_t CalibrationSliceEndBase = std::numeric_limits<uint64_t>::max();

using Sample = std::pair<uint64_t, double>; /// bytes, nanoseconds

/// Synthetic payload with roughly the redundancy of real slice state: fixed-size records whose leading
/// bytes repeat across records (keys and timestamps from a narrow range) and whose tail varies. A
/// payload of pure zeroes would make every compressor look free; a payload of pure noise would make it
/// look useless.
std::vector<std::byte> makePayload(const uint64_t bytes)
{
    std::vector<std::byte> payload(bytes);
    for (uint64_t i = 0; i < bytes; ++i)
    {
        const auto positionInRecord = i % 64;
        payload[i] = positionInRecord < 40 ? static_cast<std::byte>(positionInRecord) : static_cast<std::byte>((i * 2654435761U) >> 24);
    }
    return payload;
}

/// Least-squares fit of nanos = fixed + perByte * bytes.
StageCost fitLine(const std::vector<Sample>& samples)
{
    INVARIANT(samples.size() >= 2, "Need at least two sizes to separate the fixed cost from the per-byte cost");

    double sumX = 0;
    double sumY = 0;
    double sumXY = 0;
    double sumXX = 0;
    for (const auto& [bytes, nanos] : samples)
    {
        const auto x = static_cast<double>(bytes);
        sumX += x;
        sumY += nanos;
        sumXY += x * nanos;
        sumXX += x * x;
    }
    const auto count = static_cast<double>(samples.size());
    const double denominator = (count * sumXX) - (sumX * sumX);

    StageCost cost;
    if (denominator == 0.0)
    {
        /// Every sample was the same size, so there is no slope to recover. Attribute everything to the
        /// per-byte term: overestimating the marginal cost is the safe direction, since it makes the
        /// predictor more reluctant to reduce state it might not get back in time.
        cost.nanosPerByte = sumX > 0 ? sumY / sumX : 0.0;
        return cost;
    }

    cost.nanosPerByte = ((count * sumXY) - (sumX * sumY)) / denominator;
    cost.fixedNanos = (sumY - (cost.nanosPerByte * sumX)) / count;

    /// A non-positive slope is measurement noise -- one preempted run at a small size is enough to tilt
    /// the fit -- not a real effect, and it would let the model predict that bigger states are cheaper.
    /// Fall back to the average cost per byte, as the degenerate case above does: attributing everything
    /// to the per-byte term overestimates the marginal cost, which is the safe direction. Clamping to
    /// zero instead would price compression as free per byte, so the predictor would reduce state
    /// regardless of the deadline -- the one model this must never produce.
    if (cost.nanosPerByte <= 0.0)
    {
        cost.nanosPerByte = sumX > 0 ? sumY / sumX : 0.0;
        cost.fixedNanos = 0.0;
        return cost;
    }
    cost.fixedNanos = std::max(cost.fixedNanos, 0.0);
    return cost;
}

template <typename Operation>
double fastestNanos(const uint32_t repetitions, Operation&& operation)
{
    double best = std::numeric_limits<double>::max();
    for (uint32_t run = 0; run < repetitions; ++run)
    {
        const auto start = std::chrono::steady_clock::now();
        operation();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        best = std::min(best, static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
    }
    return best;
}

}

StateReductionCostModel
calibrateStateReduction(const CompressionAlgorithm& compression, SpillStore& spillStore, const CalibrationConfig& config)
{
    PRECONDITION(config.minBytes > 0, "Calibration needs a non-zero smallest size");
    PRECONDITION(config.maxBytes >= config.minBytes * 2, "Calibration needs at least two distinct sizes to fit a line");
    const auto repetitions = std::max(config.repetitions, 1U);

    std::vector<Sample> compressSamples;
    std::vector<Sample> decompressSamples;
    std::vector<Sample> writeSamples;
    std::vector<Sample> readSamples;

    uint64_t totalRaw = 0;
    uint64_t totalCompressed = 0;
    uint64_t calibrationSlice = CalibrationSliceEndBase;

    for (uint64_t bytes = config.minBytes; bytes <= config.maxBytes; bytes *= 2)
    {
        const auto payload = makePayload(bytes);

        std::vector<std::byte> compressed;
        compressSamples.emplace_back(bytes, fastestNanos(repetitions, [&] { compressed = compression.compress(payload); }));

        totalRaw += bytes;
        totalCompressed += compressed.size();

        decompressSamples.emplace_back(
            compressed.size(), fastestNanos(repetitions, [&] { std::ignore = compression.decompress(compressed, bytes); }));

        const auto sliceEnd = calibrationSlice--;
        writeSamples.emplace_back(bytes, fastestNanos(repetitions, [&] { spillStore.put(sliceEnd, payload); }));
        readSamples.emplace_back(bytes, fastestNanos(repetitions, [&] { std::ignore = spillStore.get(sliceEnd); }));
        spillStore.erase(sliceEnd);
    }

    StateReductionCostModel model;
    model.compress = fitLine(compressSamples);
    model.decompress = fitLine(decompressSamples);
    model.spillWrite = fitLine(writeSamples);
    model.spillRead = fitLine(readSamples);
    model.compressionRatio
        = totalRaw > 0 ? std::clamp(static_cast<double>(totalCompressed) / static_cast<double>(totalRaw), 0.01, 1.0) : 1.0;
    return model;
}

}
