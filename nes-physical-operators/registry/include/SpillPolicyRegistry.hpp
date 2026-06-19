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
#include <string>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Util/Registry.hpp>

namespace NES
{

class WatermarkPredictor;

using SpillPolicyRegistryReturnType = std::unique_ptr<SpillPolicy>;

/// Arguments forwarded to a SpillPolicy factory at create time.
struct SpillPolicyRegistryArguments
{
    double lowMemoryBound{0.6};
    double highMemoryBound{0.85};
    std::chrono::milliseconds horizon{50};
    WatermarkPredictor* predictor{nullptr};
};

class SpillPolicyRegistry
    : public BaseRegistry<SpillPolicyRegistry, std::string, SpillPolicyRegistryReturnType, SpillPolicyRegistryArguments>
{
};

}

#define INCLUDED_FROM_SPILL_POLICY_REGISTRY
#include <SpillPolicyGeneratedRegistrar.inc>
#undef INCLUDED_FROM_SPILL_POLICY_REGISTRY
