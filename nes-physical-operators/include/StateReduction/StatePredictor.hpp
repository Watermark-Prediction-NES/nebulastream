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
#include <StateReductionTypes.hpp>

namespace NES
{

/// Decides what to do with one slice's state.
///
/// The interesting question is not "are we short on memory" but "can we afford to put this state
/// somewhere cheaper and still have it back by the time it is needed". That makes this a cost model, not
/// a threshold: predict() is handed how big the state is and how long it has, and answers with the most
/// memory-reducing option whose round trip fits in that window.
///
/// Implementations are used concurrently — predict() is called from the thread that advances the build
/// watermark, update()/updateAfterRestore() from probe threads — so they must be internally synchronised.
class StatePredictor
{
public:
    virtual ~StatePredictor() = default;

    StatePredictor() = default;
    StatePredictor(const StatePredictor&) = delete;
    StatePredictor(StatePredictor&&) = delete;
    StatePredictor& operator=(const StatePredictor&) = delete;
    StatePredictor& operator=(StatePredictor&&) = delete;

    /// @param bytes resident size of the slice's state right now.
    /// @param maxRequiredTimeInMilliSeconds wall-clock budget until the state is needed again, as
    ///        predicted by the WatermarkPredictor. Zero means "needed now", which only KeepInMemory
    ///        satisfies, so a cold or absent watermark predictor never reduces anything.
    [[nodiscard]] virtual StateDecision predict(uint64_t bytes, uint64_t maxRequiredTimeInMilliSeconds) const = 0;

    /// Reports a completed reduction so the model can correct itself against the machine and the data it
    /// actually runs on, rather than staying on its start-up calibration forever.
    /// @param reducedPercentage share of `bytes` that was freed, in [0, 1].
    /// @param bytes resident size before the reduction.
    /// @param elapsed how long the reduction took.
    virtual void update(StateDecision decision, double reducedPercentage, uint64_t bytes, std::chrono::nanoseconds elapsed) = 0;

    /// Reports a completed restore. Separate from update() because a restore is the other half of the
    /// round trip predict() has to estimate, and it carries no reduction ratio — folding both into one
    /// call would mean inferring the direction from the arguments.
    /// @param reducedBytes size of the stored (possibly compressed) representation that was read back.
    virtual void updateAfterRestore(StateDecision decision, uint64_t reducedBytes, std::chrono::nanoseconds elapsed) = 0;
};

}
