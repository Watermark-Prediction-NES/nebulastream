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

#include <SliceStore/Spill/SpillPolicy.hpp>

namespace NES
{

/// Pure-pressure policy. Spill when sampled pressure is above `highBound`; otherwise keep.
/// `lowBound` is reserved for future "restore below threshold" behaviour (not implemented in v1 —
/// restores happen lazily on probe access).
class ReactiveSpillPolicy final : public SpillPolicy
{
public:
    ReactiveSpillPolicy(double lowBound, double highBound) noexcept : lowBound(lowBound), highBound(highBound) { }

    [[nodiscard]] SpillDecision decide(const SliceSpillContext& /*ctx*/, double memoryPressure) const override
    {
        if (memoryPressure >= highBound)
        {
            return SpillDecision::Spill;
        }
        return SpillDecision::Keep;
    }

private:
    [[maybe_unused]] double lowBound;
    double highBound;
};

}
