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

PredictionMetrics runBenchmark(
    WatermarkPredictor& predictor,
    const WatermarkTrace& observed,
    const WatermarkTrace& truth,
    size_t observePrefix,
    const std::vector<uint64_t>& horizons)
{
    PredictionMetrics m;
    if (observePrefix == 0 || observePrefix > observed.size())
    {
        return m;
    }
    for (size_t i = 0; i < observePrefix; ++i)
    {
        predictor.observe(observed[i].watermarkTs, observed[i].wallClock);
    }
    const auto lastSeenWatermark = observed[observePrefix - 1].watermarkTs;

    double sumAbs = 0.0;
    double sumSq = 0.0;
    double sumPct = 0.0;
    double maxErr = 0.0;
    size_t n = 0;
    size_t pctSamples = 0;
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
        const double err = static_cast<double>(predicted.getRawValue()) - static_cast<double>(trueT->getRawValue());
        const double absErr = std::abs(err);
        sumAbs += absErr;
        sumSq += err * err;
        if (absErr > maxErr)
        {
            maxErr = absErr;
        }
        /// MAPE is undefined when true == 0; skip those rather than blow up the average.
        const auto trueAbs = std::abs(static_cast<double>(trueT->getRawValue()));
        if (trueAbs > 0.0)
        {
            sumPct += (absErr / trueAbs) * 100.0;
            ++pctSamples;
        }
        ++n;
    }
    if (n == 0)
    {
        return m;
    }
    m.mae = sumAbs / static_cast<double>(n);
    m.rmse = std::sqrt(sumSq / static_cast<double>(n));
    m.mape = pctSamples == 0 ? 0.0 : sumPct / static_cast<double>(pctSamples);
    m.maxError = maxErr;
    m.samples = n;
    return m;
}

}
