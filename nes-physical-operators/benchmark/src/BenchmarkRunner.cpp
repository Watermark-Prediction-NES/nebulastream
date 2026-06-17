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

#include <BenchmarkRunner.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <Time/Timestamp.hpp>
#include <Trace/Trace.hpp>
#include <Watermark/WatermarkPredictor.hpp>
#include <PredictionMetrics.hpp>
#include <TruthOracle.hpp>

namespace NES
{

std::vector<PredictionSample> runBenchmark(
    WatermarkPredictor& predictor,
    const WatermarkTrace& observed,
    const WatermarkTrace& truth,
    size_t warmup,
    const std::vector<uint64_t>& horizons)
{
    std::vector<PredictionSample> samples;
    if (warmup == 0 || warmup > observed.size())
    {
        return samples;
    }

    /// Batch warm-up: observe the prefix without scoring so the predictor reaches steady state.
    for (size_t i = 0; i < warmup; ++i)
    {
        predictor.observe(observed[i].watermarkTs, observed[i].wallClock);
    }

    /// Prequential phase: observe one sample, then query every horizon. A horizon whose crossing
    /// falls past the end of the truth trace yields no ground truth and is skipped (so eval ticks
    /// near the end naturally contribute fewer horizons).
    for (size_t i = warmup; i < observed.size(); ++i)
    {
        predictor.observe(observed[i].watermarkTs, observed[i].wallClock);
        const auto lastSeenWatermark = observed[i].watermarkTs;
        for (const auto h : horizons)
        {
            const Timestamp target{lastSeenWatermark.getRawValue() + h};
            const auto trueT = trueWallClockForTarget(truth, target);
            if (!trueT)
            {
                continue;
            }
            const auto predicted = predictor.predictWallClock(target);
            if (predicted.getRawValue() == Timestamp::INVALID_VALUE)
            {
                continue;
            }
            const double signedErr = static_cast<double>(predicted.getRawValue()) - static_cast<double>(trueT->getRawValue());
            samples.push_back(PredictionSample{
                .evalOffset = i - warmup,
                .horizon = h,
                .absErr = std::abs(signedErr),
                .signedErr = signedErr,
                .trueWall = static_cast<double>(trueT->getRawValue())});
        }
    }
    return samples;
}

}
