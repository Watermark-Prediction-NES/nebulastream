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

#include <Watermark/EwmaWatermarkPredictor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Time/Timestamp.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

namespace
{
constexpr double MIN_RATE = 1e-12;
}

EwmaWatermarkPredictor::EwmaWatermarkPredictor(const double alpha) : smoothingFactor{alpha}
{
    INVARIANT(alpha > 0.0 && alpha <= 1.0, "EWMA alpha must be in (0, 1], got {}", alpha);
}

void EwmaWatermarkPredictor::observe(const Timestamp watermarkTs, const Timestamp wallClock)
{
    if (!hasFirstObservation)
    {
        lastWatermark = watermarkTs;
        lastWallClock = wallClock;
        hasFirstObservation = true;
        return;
    }
    if (wallClock <= lastWallClock)
    {
        /// Same or out-of-order wall-clock sample: skip to keep dt > 0.
        return;
    }
    const auto watermarkDelta = static_cast<double>(watermarkTs.getRawValue()) - static_cast<double>(lastWatermark.getRawValue());
    const auto wallClockDelta = static_cast<double>(wallClock.getRawValue()) - static_cast<double>(lastWallClock.getRawValue());
    const double sampleRate = watermarkDelta / wallClockDelta;
    rateEstimate = hasRateEstimate ? (smoothingFactor * sampleRate + (1.0 - smoothingFactor) * rateEstimate) : sampleRate;
    hasRateEstimate = true;
    lastWatermark = watermarkTs;
    lastWallClock = wallClock;
}

Timestamp EwmaWatermarkPredictor::predictWallClock(const Timestamp target) const
{
    if (!hasRateEstimate)
    {
        return Timestamp{Timestamp::INVALID_VALUE};
    }
    if (target <= lastWatermark)
    {
        return lastWallClock;
    }
    const double gap = static_cast<double>(target.getRawValue()) - static_cast<double>(lastWatermark.getRawValue());
    const double safeRate = std::max(rateEstimate, MIN_RATE);
    const double predicted = static_cast<double>(lastWallClock.getRawValue()) + gap / safeRate;
    /// Saturate to the largest representable finite Timestamp; INVALID_VALUE is reserved for "no state".
    constexpr auto saturate = static_cast<double>(Timestamp::INVALID_VALUE - 1);
    if (!std::isfinite(predicted) || predicted >= saturate)
    {
        return Timestamp{Timestamp::INVALID_VALUE - 1};
    }
    return Timestamp{static_cast<Timestamp::Underlying>(predicted)};
}

}
