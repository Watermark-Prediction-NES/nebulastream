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
#include <StateReduction/StatePredictor.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

/// Always answers with the same leaf, whatever the size or the deadline.
///
/// This is how a benchmark isolates one leaf: forcing every slice down the same branch turns the
/// measurement into "what does compressing/spilling cost", with the prediction quality held constant.
/// Forcing KeepInMemory is also the in-memory baseline every other variant is compared against, so there
/// is no separate do-nothing implementation.
class ForcedStatePredictor final : public StatePredictor
{
public:
    explicit ForcedStatePredictor(const StateDecision decision) : decision(decision) { }

    [[nodiscard]] StateDecision predict(uint64_t /*bytes*/, uint64_t /*maxRequiredTimeInMilliSeconds*/) const override { return decision; }

    /// Deliberately inert: a forced predictor that learned would stop being a fixed control.
    void update(StateDecision, double, uint64_t, std::chrono::nanoseconds) override { }

    void updateAfterRestore(StateDecision, uint64_t, std::chrono::nanoseconds) override { }

private:
    StateDecision decision;
};

}
