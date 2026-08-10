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
#include <chrono>
#include <cstdint>
#include <memory>
#include <string_view>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Time/Timestamp.hpp>
#include <Watermark/EwmaWatermarkPredictor.hpp>

namespace NES
{

/// Pressure-driven spill policy. Spills a slice when sampled pressure is at/above `highBound`.
///
/// Optionally refined by a watermark predictor: when a predictor is present and warm, a slice whose
/// estimated trigger is within `horizon` is kept resident even under pressure, so the probe doesn't
/// pay restore latency on an imminent trigger. With no predictor (or a cold one returning
/// INVALID_VALUE) the policy is pure-pressure — i.e. the old "reactive" behaviour.
///
/// Predictor selection (`predictorName`): "ewma", "kalman", or "robustkalman"; unknown values fall
/// back to "ewma" and log a `NES_WARN`.
class PressureSpillPolicy final : public SpillPolicy
{
public:
    /// Pure-pressure (reactive): no predictor.
    explicit PressureSpillPolicy(double highBound) noexcept;

    /// Predictor-refined (predictive): consult the predictor within `horizon` before spilling.
    PressureSpillPolicy(double highBound, std::chrono::milliseconds horizon, std::string_view predictorName) noexcept;

    [[nodiscard]] SpillDecision decide(const SliceSpillContext& ctx, double memoryPressure) const override;

    void observe(Timestamp now, Timestamp globalWatermark) noexcept override;

    [[nodiscard]] SpillPolicyStatistics stats() const noexcept override;

private:
    double highBound;
    std::chrono::milliseconds horizon{0};
    std::unique_ptr<WatermarkPredictor> predictor; /// null => pure-pressure (reactive)
    Timestamp lastObservedWatermark{Timestamp::INVALID_VALUE};

    /// Decision counters. Mutable + atomic because decide() is const and runs on worker threads;
    /// they are pure telemetry and guard no other state, so relaxed ordering suffices.
    mutable std::atomic<uint64_t> keptByPrediction{0};
    mutable std::atomic<uint64_t> spilledPredictorCold{0};
    mutable std::atomic<uint64_t> spilledBeyondHorizon{0};
};

}
