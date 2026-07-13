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

/// Configuration for the out-of-order noise model. Lives next to the model itself so adding new
/// models doesn't require touching shared headers.
///
/// Semantics: this model perturbs the ORDER in which watermark *values* arrive, not their
/// wall-clock delivery times. In a real cloud-edge stream, tuples emitted in event-time order can
/// still be delivered out of order (network/scheduling variance, multi-partition merges), so the
/// watermark-value sequence the predictor observes contains bounded local reordering while
/// wall-clock keeps advancing on its normal schedule. This is a genuinely different failure mode
/// from GaussianNoiseModel's wall-clock jitter/lateness, which never regresses the watermark value
/// itself -- it directly exercises the watermark-regression guard every predictor's observe() has
/// to reject on (see EwmaWatermarkPredictor::observe() etc.: `watermarkTs < lastWatermark`).
struct OutOfOrderNoiseConfig
{
    /// Probability (in [0, 1]) that the watermark value at a given position is displaced forward by
    /// a random amount. Default 0 = perfectly in order.
    double reorderProbability{0.0};

    /// Maximum number of positions a displaced watermark value may be pushed back by (bounded local
    /// reordering). Larger values allow deeper regressions. Must be >= 1 to have any effect.
    uint64_t maxDelay{1};

    /// RNG seed for reproducibility.
    uint64_t seed{42};
};

/// Applies bounded local reordering to the watermark-value column of a clean trace, keeping the
/// wall-clock delivery schedule untouched. The output watermark column is a permutation of the
/// input's (no values lost or invented), each displaced by at most a few positions, so the
/// predictor sees the same values -- just occasionally out of order, forcing observe() onto its
/// out-of-order/regression path.
class OutOfOrderNoiseModel final : public NoiseModel
{
public:
    explicit OutOfOrderNoiseModel(OutOfOrderNoiseConfig cfg);
    [[nodiscard]] WatermarkTrace apply(const WatermarkTrace& clean) const override;

private:
    OutOfOrderNoiseConfig cfg;
};

}
