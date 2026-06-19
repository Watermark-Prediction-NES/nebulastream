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

#include <chrono>
#include <memory>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Time/Timestamp.hpp>
#include <Watermark/EwmaWatermarkPredictor.hpp>

namespace NES
{

/// Predictive spill policy: spills a slice when (a) pressure exceeds the high threshold AND
/// (b) the watermark predictor estimates the slice will NOT be triggered within `horizon`.
/// Owns an EwmaWatermarkPredictor and feeds it via observe().
///
/// Cost-model fallback: in this v1 implementation we do not yet maintain a rolling-window cost
/// model; the policy falls back to reactive behaviour (pressure-only) when the predictor returns
/// INVALID_VALUE (insufficient state). The plan §4.B7 mitigation hook is reserved for a future PR.
class PredictiveSpillPolicy final : public SpillPolicy
{
public:
    PredictiveSpillPolicy(double lowBound, double highBound, std::chrono::milliseconds horizon) noexcept;

    [[nodiscard]] SpillDecision decide(const SliceSpillContext& ctx, double memoryPressure) const override;

    void observe(Timestamp now, Timestamp globalWatermark) noexcept override;

private:
    [[maybe_unused]] double lowBound;
    double highBound;
    std::chrono::milliseconds horizon;
    std::unique_ptr<WatermarkPredictor> predictor;
    Timestamp lastObservedWatermark{Timestamp::INVALID_VALUE};
};

}
