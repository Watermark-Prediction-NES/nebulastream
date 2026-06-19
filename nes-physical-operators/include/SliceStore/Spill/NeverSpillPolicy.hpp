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

/// Default policy: always returns Keep. Equivalent to "spill disabled" for any slice the decorator
/// is wrapping. Used as the fallback when SpillConfiguration.enabled is true but no specific policy
/// was selected, and as the default policy registered under name "never".
class NeverSpillPolicy final : public SpillPolicy
{
public:
    NeverSpillPolicy() = default;

    [[nodiscard]] SpillDecision decide(const SliceSpillContext& /*ctx*/, double /*memoryPressure*/) const override
    {
        return SpillDecision::Keep;
    }
};

}
