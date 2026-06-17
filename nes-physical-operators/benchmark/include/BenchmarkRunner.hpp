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

#include <cstddef>
#include <cstdint>
#include <vector>
#include <Trace/Trace.hpp>
#include <Watermark/WatermarkPredictor.hpp>
#include <PredictionMetrics.hpp>

namespace NES
{

/// Rolling / prequential evaluation. Batch-observes the first `warmup` samples (no scoring) so the
/// predictor reaches steady state, then walks the rest of the trace one sample at a time: observe,
/// then immediately query every horizon and score the prediction against the truth trace. Emits one
/// PredictionSample per (eval tick, horizon) whose crossing actually falls within the truth trace.
///
/// observed: what the predictor sees (may be noisy).
/// truth:    the clean trace used to compute ground-truth wall-clock crossings.
///           Pass the same trace as observed if you don't apply noise.
[[nodiscard]] std::vector<PredictionSample> runBenchmark(
    WatermarkPredictor& predictor,
    const WatermarkTrace& observed,
    const WatermarkTrace& truth,
    size_t warmup,
    const std::vector<uint64_t>& horizons);

}
