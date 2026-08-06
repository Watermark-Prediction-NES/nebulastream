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
#include <cstdint>
#include <mutex>
#include <StateReduction/StatePredictor.hpp>
#include <StateReduction/StateReductionCostModel.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

/// Picks the leaf that frees the most memory while still fitting in the time available.
///
/// Both halves of the decision are measured rather than configured. How long a reduction takes comes
/// from StateReductionCalibrator at operator start-up and is corrected at runtime by update() /
/// updateAfterRestore(); how long we have comes from the WatermarkPredictor. That is what removes the
/// hand-tuned time horizons a threshold policy needs — horizons that cannot transfer between queries
/// anyway, because the right value depends on the window slide.
class CostModelStatePredictor final : public StatePredictor
{
public:
    /// @param learningRate weight given to each new observation when correcting the model, in [0, 1].
    ///        0 freezes the calibration, 1 throws it away on every sample.
    explicit CostModelStatePredictor(StateReductionCostModel initialModel, double learningRate);

    [[nodiscard]] StateDecision predict(uint64_t bytes, uint64_t maxRequiredTimeInMilliSeconds) const override;

    void update(StateDecision decision, double reducedPercentage, uint64_t bytes, std::chrono::nanoseconds elapsed) override;

    void updateAfterRestore(StateDecision decision, uint64_t reducedBytes, std::chrono::nanoseconds elapsed) override;

    /// Snapshot for tests and for logging what the model converged on.
    [[nodiscard]] StateReductionCostModel currentModel() const;

private:
    mutable std::mutex modelMutex;
    StateReductionCostModel model;
    double learningRate;
};

}
