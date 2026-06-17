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

#include <Noise/GaussianNoiseModel.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <Time/Timestamp.hpp>
#include <Trace/Trace.hpp>

namespace NES
{

GaussianNoiseModel::GaussianNoiseModel(GaussianNoiseConfig cfg) : cfg{cfg}
{
}

WatermarkTrace GaussianNoiseModel::apply(const WatermarkTrace& clean) const
{
    WatermarkTrace noisy;
    noisy.reserve(clean.size());

    std::mt19937_64 rng{cfg.seed};
    std::normal_distribution<double> jitter{0.0, cfg.wallClockStddev};
    std::bernoulli_distribution late{std::clamp(cfg.lateProbability, 0.0, 1.0)};
    std::normal_distribution<double> extraDelay{cfg.lateExtraDelayMean, cfg.lateExtraDelayStddev};

    for (const auto& sample : clean)
    {
        auto wallD = static_cast<double>(sample.wallClock.getRawValue());
        if (cfg.wallClockStddev > 0.0)
        {
            wallD += jitter(rng);
        }
        if (cfg.lateProbability > 0.0 && late(rng))
        {
            wallD += std::abs(extraDelay(rng));
        }
        if (wallD < 1.0)
        {
            wallD = 1.0;
        }
        noisy.push_back({sample.watermarkTs, Timestamp{static_cast<Timestamp::Underlying>(wallD)}});
    }
    return noisy;
}

}
