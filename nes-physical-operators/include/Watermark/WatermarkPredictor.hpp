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
#include <memory>
#include <Time/Timestamp.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

/// Predicts the wall-clock time at which the watermark will reach a target event-time.
/// observe() is stateful, predictWallClock() is a pure read over the internal state.
/// Implementations are not required to be thread-safe.
class WatermarkPredictor
{
public:
    virtual ~WatermarkPredictor() = default;

    virtual void observe(Timestamp watermarkTs, Timestamp wallClock) = 0;

    /// Returns INVALID_VALUE only when the predictor has insufficient state.
    [[nodiscard]] virtual Timestamp predictWallClock(Timestamp target) const = 0;

    /// Returns nullptr for NONE, which leaves every slice looking immediately needed and so keeps
    /// state reduction switched off however cheap a reduction looks.
    [[nodiscard]] static std::unique_ptr<WatermarkPredictor> create(WatermarkPredictorType type);
};

}
