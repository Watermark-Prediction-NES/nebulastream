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

#include <Noise/OutOfOrderNoiseModel.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>
#include <Trace/Trace.hpp>

namespace NES
{

OutOfOrderNoiseModel::OutOfOrderNoiseModel(OutOfOrderNoiseConfig cfg) : cfg{cfg}
{
}

WatermarkTrace OutOfOrderNoiseModel::apply(const WatermarkTrace& clean) const
{
    /// The wall-clock column is the delivery schedule and stays exactly as generated; only the
    /// watermark-value column is reordered. Start from a copy so a no-op config (probability 0 or
    /// maxDelay 0) returns the clean trace unchanged.
    WatermarkTrace noisy = clean;
    const std::size_t n = noisy.size();
    if (n < 2 || cfg.reorderProbability <= 0.0 || cfg.maxDelay == 0)
    {
        return noisy;
    }

    /// Displace watermark values forward by a bounded amount via windowed swaps. Swapping the value
    /// at i with a later value at j = i + d (d in [1, maxDelay]) moves the (smaller) value at i to a
    /// later position, so it now follows larger values there -- a genuine watermark regression the
    /// predictor's observe() must reject. The wall-clock at each position is left in place, so the
    /// predictor sees the reordered value paired with its slot's normal arrival time.
    ///
    /// frozen[j] is set after each swap so the displaced value at j cannot itself become a source
    /// for another swap, which would move it arbitrarily far beyond maxDelay (cascaded displacement).
    /// Symmetrically, a position that was already a target (frozen[i]) is skipped as a source.
    std::mt19937_64 rng{cfg.seed};
    std::bernoulli_distribution reorder{std::clamp(cfg.reorderProbability, 0.0, 1.0)};
    std::uniform_int_distribution<std::uint64_t> delay{1, cfg.maxDelay};

    std::vector<bool> frozen(n, false);
    for (std::size_t i = 0; i + 1 < n; ++i)
    {
        if (frozen[i] || !reorder(rng))
        {
            continue;
        }
        const std::size_t j = std::min(i + static_cast<std::size_t>(delay(rng)), n - 1);
        if (frozen[j])
        {
            continue;
        }
        std::swap(noisy[i].watermarkTs, noisy[j].watermarkTs);
        frozen[j] = true;
    }
    return noisy;
}

}
