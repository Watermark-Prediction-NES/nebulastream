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

#include <cstdint>
#include <Noise/NoiseModel.hpp>
#include <Trace/Trace.hpp>

namespace NES
{

/// Configuration for the Gaussian noise model. Lives next to the model itself so adding new
/// models doesn't require touching shared headers.
///
/// Semantics: noise is applied to the wall-clock arrival of each watermark only. Watermark values
/// are emitted by upstream sources and are not perturbed -- a watermark value of W reached at clean
/// wall-clock T is *seen by the predictor* at T + jitter [+ extra lateness if sampled].
struct GaussianNoiseConfig
{
    /// Gaussian std dev on the wall-clock arrival of every watermark. Default 0 = no jitter.
    double wallClockStddev{0.0};

    /// Probability (in [0, 1]) that a watermark is selected as "late" and gets an extra delay
    /// beyond the baseline Gaussian jitter. Default 0 = no late watermarks.
    double lateProbability{0.0};

    /// Mean of the extra Gaussian delay applied to a late-selected watermark (wall-clock units).
    double lateExtraDelayMean{0.0};

    /// Std dev of the extra Gaussian delay applied to a late-selected watermark.
    double lateExtraDelayStddev{0.0};

    /// RNG seed for reproducibility.
    uint64_t seed{42};
};

/// Adds Gaussian wall-clock jitter and an optional Bernoulli straggler spike to a clean trace.
class GaussianNoiseModel final : public NoiseModel
{
public:
    explicit GaussianNoiseModel(GaussianNoiseConfig cfg);
    [[nodiscard]] WatermarkTrace apply(const WatermarkTrace& clean) const override;

private:
    GaussianNoiseConfig cfg;
};

}
