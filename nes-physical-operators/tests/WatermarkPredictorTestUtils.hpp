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
#include <Time/Timestamp.hpp>
#include <Watermark/WatermarkPredictor.hpp>

namespace NES
{

constexpr uint64_t INVALID_TS = Timestamp::INVALID_VALUE;

constexpr Timestamp T(const uint64_t v)
{
    return Timestamp{v};
}

/// Final (watermark, wall-clock) point of a constant-rate feed, useful for predicate assertions.
struct StreamEndpoint
{
    uint64_t lastWatermark;
    uint64_t lastWallClock;
};

/// Feeds a constant-rate watermark stream into a predictor.
/// Each tick: watermark += eventStep, wall += wallStep.
inline StreamEndpoint feedConstantRate(
    WatermarkPredictor& p,
    const uint64_t ticks,
    const uint64_t eventStep,
    const uint64_t wallStep,
    uint64_t startW = 1000,
    uint64_t startT = 1000)
{
    uint64_t w = startW;
    uint64_t t = startT;
    for (uint64_t i = 0; i < ticks; ++i)
    {
        p.observe(T(w), T(t));
        w += eventStep;
        t += wallStep;
    }
    return {w - eventStep, t - wallStep};
}

}
