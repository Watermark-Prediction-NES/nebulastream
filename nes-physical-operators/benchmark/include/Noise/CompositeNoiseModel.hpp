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

#include <functional>
#include <vector>
#include <Noise/NoiseModel.hpp>
#include <Trace/Trace.hpp>

namespace NES
{

/// Chains several noise models, applying each to the output of the previous one (in listed order).
///
/// Wall-clock jitter (GaussianNoiseModel) and out-of-order arrival (OutOfOrderNoiseModel) are
/// independent axes in practice -- both stem from network/scheduling variance -- so a real stream
/// can exhibit both at once. This composes them without either model having to know about the other.
///
/// The component models are held by (non-owning) reference: apply() is called synchronously while
/// the referenced models are still alive (see makeNoisyExperiment), so no ownership is needed.
class CompositeNoiseModel final : public NoiseModel
{
public:
    explicit CompositeNoiseModel(std::vector<std::reference_wrapper<const NoiseModel>> models);
    [[nodiscard]] WatermarkTrace apply(const WatermarkTrace& clean) const override;

private:
    std::vector<std::reference_wrapper<const NoiseModel>> models;
};

}
