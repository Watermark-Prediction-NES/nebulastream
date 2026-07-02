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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <Noise/GaussianNoiseModel.hpp>
#include <Time/Timestamp.hpp>
#include <Trace/PiecewiseConstantTraceSource.hpp>
#include <Watermark/EwmaWatermarkPredictor.hpp>
#include <Watermark/KalmanWatermarkPredictor.hpp>
#include <Watermark/MlpWatermarkPredictor.hpp>
#include <Watermark/NeuralKalmanWatermarkPredictor.hpp>
#include <Watermark/RobustAdaptiveKalmanWatermarkPredictor.hpp>
#include <Watermark/WatermarkPredictor.hpp>
#include <BenchmarkRunner.hpp>
#include <Experiment.hpp>
#include <PredictionMetrics.hpp>

namespace NES
{

/// Predictor instances are built on demand so each experiment starts with a fresh state.
struct PredictorEntry
{
    std::string name;
    std::function<std::unique_ptr<WatermarkPredictor>()> make;
};

namespace
{

/// Each (trace x predictor) cell times PredictQueriesPerRep predictWallClock() calls, repeated
/// Repetitions times. PredictQueriesPerRep is fixed and tuned so one rep ~1s on a modern x86 core
/// (e.g. amd7950x3d); lower it for slower hosts. Targets sweep a power-of-two span just past the last
/// observed watermark so every call stays on the extrapolation path with only a cheap bit-mask.
constexpr std::size_t Repetitions = 5;
constexpr std::size_t PredictQueriesPerRep = 100'000'000;
constexpr std::uint64_t TargetSpan = 1ULL << 20;

/// observe() is far more expensive than predictWallClock() for the ML predictors (a full
/// forward+backward pass vs. a closed-form extrapolation), so it gets its own, much smaller op
/// budget -- tuned so the heaviest predictor (MLP) still finishes a rep in roughly a second.
constexpr std::size_t ObserveCallsPerRep = 2'000'000;

/// Per-cell timing of one predictor operation (predictWallClock() or observe()), aggregated over
/// Repetitions reps. Reports both latency (ns/op) and its reciprocal, throughput (ops/sec), so
/// downstream consumers don't have to invert one to get the other.
struct TimingStats
{
    std::size_t reps{0};
    double meanMs{0.0};
    double minMs{0.0};
    double nsPerOp{0.0};
    double opsPerSec{0.0};
};

/// Times predictWallClock() only: the predictor is driven through the same observed prefix the
/// accuracy run uses (untimed), then a fixed batch of queries is timed. Deterministic accuracy is
/// unaffected; this just measures prediction cost.
TimingStats timePredict(const PredictorEntry& entry, const Experiment& exp)
{
    TimingStats stats{.reps = Repetitions};
    if (exp.warmup == 0 || exp.warmup > exp.observed.size())
    {
        return stats;
    }

    std::uint64_t sink = 0;
    double sumMs = 0.0;
    double minMs = std::numeric_limits<double>::max();
    for (std::size_t rep = 0; rep < Repetitions; ++rep)
    {
        auto predictor = entry.make();
        for (std::size_t i = 0; i < exp.warmup; ++i)
        {
            predictor->observe(exp.observed[i].watermarkTs, exp.observed[i].wallClock);
        }
        const std::uint64_t base = exp.observed[exp.warmup - 1].watermarkTs.getRawValue() + 1;

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < PredictQueriesPerRep; ++i)
        {
            const Timestamp target{base + (i & (TargetSpan - 1))};
            sink += predictor->predictWallClock(target).getRawValue();
        }
        const auto end = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        sumMs += ms;
        minMs = std::min(minMs, ms);
    }

    /// Defeat dead-code elimination of the timed loop (predictWallClock is pure, so without an
    /// observable use of its results the whole loop could be optimised away).
    static volatile std::uint64_t timingSink = 0;
    timingSink = timingSink + sink;

    stats.meanMs = sumMs / static_cast<double>(Repetitions);
    stats.minMs = minMs;
    stats.nsPerOp = (stats.meanMs * 1e6) / static_cast<double>(PredictQueriesPerRep);
    stats.opsPerSec = (stats.nsPerOp > 0.0) ? (1e9 / stats.nsPerOp) : 0.0;
    return stats;
}

/// Times observe() only: this is the per-tuple ingestion path (every incoming watermark update
/// calls it, see DefaultTimeBasedSliceStore/PressureSpillPolicy), so its throughput -- not just
/// predictWallClock()'s latency -- bounds how fast a pipeline can feed the predictor. The predictor
/// is warmed up identically to the accuracy run (untimed), then fed a synthetic strictly-increasing
/// stream (rate 1.0) so every timed call takes the real "new observation" path rather than the cheap
/// out-of-order/duplicate rejection early-out.
TimingStats timeObserve(const PredictorEntry& entry, const Experiment& exp)
{
    TimingStats stats{.reps = Repetitions};
    if (exp.warmup == 0 || exp.warmup > exp.observed.size())
    {
        return stats;
    }

    double sumMs = 0.0;
    double minMs = std::numeric_limits<double>::max();
    for (std::size_t rep = 0; rep < Repetitions; ++rep)
    {
        auto predictor = entry.make();
        for (std::size_t i = 0; i < exp.warmup; ++i)
        {
            predictor->observe(exp.observed[i].watermarkTs, exp.observed[i].wallClock);
        }
        const std::uint64_t baseWatermark = exp.observed[exp.warmup - 1].watermarkTs.getRawValue();
        const std::uint64_t baseWallClock = exp.observed[exp.warmup - 1].wallClock.getRawValue();

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 1; i <= ObserveCallsPerRep; ++i)
        {
            predictor->observe(Timestamp{baseWatermark + i}, Timestamp{baseWallClock + i});
        }
        const auto end = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        sumMs += ms;
        minMs = std::min(minMs, ms);
    }

    stats.meanMs = sumMs / static_cast<double>(Repetitions);
    stats.minMs = minMs;
    stats.nsPerOp = (stats.meanMs * 1e6) / static_cast<double>(ObserveCallsPerRep);
    stats.opsPerSec = (stats.nsPerOp > 0.0) ? (1e9 / stats.nsPerOp) : 0.0;
    return stats;
}

/// The distinct clean scenario shapes. Defined once so buildExperiments() and printTraces() agree.
/// Each is long enough past the warm-up boundary that the largest horizon still resolves against the
/// truth trace for most eval ticks.
struct BaseTraces
{
    PiecewiseConstantTraceSource constantFast{{{300, 100, 50}}};

    /// Stall: rate 2.0, then the watermark holds flat (eventStep 0) while wall-clock keeps advancing,
    /// then resumes at 2.0. Warm-up ends at the stall onset so the rolling eval spans the whole stall.
    PiecewiseConstantTraceSource stall{{{100, 100, 50}, {100, 0, 50}, {100, 100, 50}}};

    /// Catch-up: rate 2.0, a shorter stall, a fast burst (rate 8.0) that races to catch up the
    /// event-time backlog, then settles back to rate 2.0. Breaks extrapolators that carried the
    /// stalled rate forward, and the final settle catches ones that overshoot on the burst rate.
    PiecewiseConstantTraceSource catchUp{{{100, 100, 50}, {50, 0, 50}, {100, 400, 50}, {100, 100, 50}}};
};

std::vector<Experiment> buildExperiments()
{
    const BaseTraces base;
    const auto& constantFast = base.constantFast;
    const auto& stall = base.stall;
    const auto& catchUp = base.catchUp;

    /// Rolling eval scores after every post-warm-up sample, so a single experiment per scenario now
    /// yields the whole error-vs-time curve (no more +N prefix sweeps). Near / mid / far horizons keep
    /// look-ahead sensitivity as its own axis. Warm-up ends at the regime onset for the event traces.
    const std::vector<uint64_t> horizons{500, 2000, 5000};

    std::vector<Experiment> experiments;

    /// Clean baselines.
    experiments.push_back(makeCleanExperiment("ConstantRate(2.0) clean", constantFast, 100, horizons));
    experiments.push_back(makeCleanExperiment("Stall(2.0->0) clean", stall, 100, horizons));
    experiments.push_back(makeCleanExperiment("CatchUp(2.0->8.0) clean", catchUp, 100, horizons));

    /// Gaussian jitter, no extra-late spikes.
    const GaussianNoiseModel mildJitter{{.wallClockStddev = 10.0, .seed = 1}};
    experiments.push_back(makeNoisyExperiment("ConstantRate(2.0) mild jitter (sd=10)", constantFast, mildJitter, 100, horizons));
    const GaussianNoiseModel heavyJitter{{.wallClockStddev = 30.0, .seed = 2}};
    experiments.push_back(makeNoisyExperiment("ConstantRate(2.0) heavy jitter (sd=30)", constantFast, heavyJitter, 100, horizons));

    /// Same jitter applied to the event traces: the stall / catch-up transitions must now be tracked
    /// through noisy arrivals (the clean versions above isolate the regime change itself).
    experiments.push_back(makeNoisyExperiment("Stall(2.0->0) mild jitter (sd=10)", stall, mildJitter, 100, horizons));
    experiments.push_back(makeNoisyExperiment("Stall(2.0->0) heavy jitter (sd=30)", stall, heavyJitter, 100, horizons));
    experiments.push_back(makeNoisyExperiment("CatchUp(2.0->8.0) mild jitter (sd=10)", catchUp, mildJitter, 100, horizons));
    experiments.push_back(makeNoisyExperiment("CatchUp(2.0->8.0) heavy jitter (sd=30)", catchUp, heavyJitter, 100, horizons));

    /// Stragglers: spike distribution fixed (mean=400, sd=100); sweep over late-fraction.
    const GaussianNoiseModel stragglersMild{
        {.wallClockStddev = 5.0, .lateProbability = 0.10, .lateExtraDelayMean = 200.0, .lateExtraDelayStddev = 50.0, .seed = 3}};
    experiments.push_back(makeNoisyExperiment("ConstantRate(2.0) + 10% stragglers", constantFast, stragglersMild, 100, horizons));

    for (const auto [lateProb, seed] : std::vector<std::pair<double, uint64_t>>{{0.30, 4}, {0.50, 5}, {0.70, 6}, {0.90, 7}})
    {
        const GaussianNoiseModel sweep{
            {.wallClockStddev = 5.0,
             .lateProbability = lateProb,
             .lateExtraDelayMean = 400.0,
             .lateExtraDelayStddev = 100.0,
             .seed = seed}};
        const auto pctLabel = std::to_string(static_cast<int>(lateProb * 100.0));
        experiments.push_back(
            makeNoisyExperiment("ConstantRate(2.0) + " + pctLabel + "% heavy stragglers", constantFast, sweep, 100, horizons));
    }

    return experiments;
}

std::vector<PredictorEntry> buildPredictors()
{
    KalmanWatermarkPredictor::Config config1{
        .processNoiseWatermark = 1, .processNoiseRate = 1e-4, .observationNoise = 1.0, .initialRateVariance = 1e6};
    KalmanWatermarkPredictor::Config config2{
        .processNoiseWatermark = 1, .processNoiseRate = 1e-1, .observationNoise = 100, .initialRateVariance = 1e6};

    return {
        {.name = "EWMA(alpha=0.3)", .make = [] { return std::make_unique<EwmaWatermarkPredictor>(0.3); }},
        {.name = "EWMA(alpha=0.5)", .make = [] { return std::make_unique<EwmaWatermarkPredictor>(0.5); }},
        {.name = "EWMA(alpha=1.0)", .make = [] { return std::make_unique<EwmaWatermarkPredictor>(1.0); }},
        {.name = "Kalman(stable)", .make = [config = config1] { return std::make_unique<KalmanWatermarkPredictor>(config); }},
        {.name = "Kalman(reactive)", .make = [config = config2] { return std::make_unique<KalmanWatermarkPredictor>(config); }},
        {.name = "RobustAdaptiveKalman", .make = [] { return std::make_unique<RobustAdaptiveKalmanWatermarkPredictor>(); }},
        {.name = "MLP(win=16,h=16)", .make = [] { return std::make_unique<MlpWatermarkPredictor>(); }},
        {.name = "NeuralKalman(h=8)", .make = [] { return std::make_unique<NeuralKalmanWatermarkPredictor>(); }},
    };
}

/// Emit the clean scenario shapes as `TRACE,scenario,event_time,wall_clock` rows for the plotting
/// notebook. benchmark.py splits these out of stdout into traces.csv; the accuracy-table parser
/// ignores any line that isn't a data row.
void printTraces()
{
    const BaseTraces base;
    const auto dump = [](const std::string& name, const PiecewiseConstantTraceSource& src)
    {
        for (const auto& sample : src.generate())
        {
            std::cout << "TRACE," << name << "," << sample.watermarkTs.getRawValue() << "," << sample.wallClock.getRawValue() << '\n';
        }
    };
    dump("ConstantRate(2.0)", base.constantFast);
    dump("Stall(2.0->0)", base.stall);
    dump("CatchUp(2.0->8.0)", base.catchUp);
}

/// One CSV row per scored prediction. Timing (predict + observe, latency and throughput) is
/// denormalised onto every row (cell-level, constant per trace x predictor) so the notebook needs
/// a single results.csv with no join. benchmark.py routes ROW lines into it. Trace and predictor
/// names contain no commas, so a plain split parses cleanly downstream.
void printRows(
    const std::string& traceName,
    const std::string& predictorName,
    const std::vector<PredictionSample>& samples,
    const TimingStats& predictTiming,
    const TimingStats& observeTiming)
{
    for (const auto& s : samples)
    {
        std::cout << "ROW," << traceName << "," << predictorName << "," << s.evalOffset << "," << s.horizon << "," << s.absErr << ","
                  << s.signedErr << "," << s.trueWall << "," << predictTiming.nsPerOp << "," << predictTiming.opsPerSec << ","
                  << observeTiming.nsPerOp << "," << observeTiming.opsPerSec << '\n';
    }
}

/// Compact "Xm YYs" / "Ys" duration for the progress line.
std::string humanSeconds(double seconds)
{
    const auto total = static_cast<long>(seconds + 0.5);
    std::ostringstream os;
    if (total >= 60)
    {
        os << (total / 60) << "m" << std::setw(2) << std::setfill('0') << (total % 60) << "s";
    }
    else
    {
        os << total << "s";
    }
    return os.str();
}

}

int run()
{
    const auto experiments = buildExperiments();
    const auto predictors = buildPredictors();
    const std::size_t totalCells = experiments.size() * predictors.size();

    /// Progress goes to stderr so it stays out of the stdout CSV stream the Python wrapper parses.
    std::cerr << "[bench] " << totalCells << " cells (" << experiments.size() << " traces x " << predictors.size() << " predictors), "
              << Repetitions << " timed reps each\n";

    /// Fixed precision keeps wall-clock magnitudes and small errors both legible in the CSV; the
    /// integer columns (eval_offset, horizon) are unaffected. sync_with_stdio off for bulk row output.
    std::ios::sync_with_stdio(false);
    std::cout << std::fixed << std::setprecision(3);

    printTraces();
    const auto runStart = std::chrono::steady_clock::now();
    std::size_t cell = 0;
    for (const auto& exp : experiments)
    {
        for (const auto& predictor : predictors)
        {
            const auto cellStart = std::chrono::steady_clock::now();
            auto p = predictor.make();
            const auto samples = runBenchmark(*p, exp.observed, exp.truth, exp.warmup, exp.horizons);
            const auto predictTiming = timePredict(predictor, exp);
            const auto observeTiming = timeObserve(predictor, exp);
            printRows(exp.name, predictor.name, samples, predictTiming, observeTiming);
            ++cell;

            const auto now = std::chrono::steady_clock::now();
            const double cellS = std::chrono::duration<double>(now - cellStart).count();
            const double elapsedS = std::chrono::duration<double>(now - runStart).count();
            const double etaS = (elapsedS / static_cast<double>(cell)) * static_cast<double>(totalCells - cell);

            std::ostringstream line;
            line << "[bench] " << cell << "/" << totalCells << " (" << std::fixed << std::setprecision(0)
                 << (100.0 * static_cast<double>(cell) / static_cast<double>(totalCells)) << "%)"
                 << " elapsed=" << humanSeconds(elapsedS) << " eta=" << humanSeconds(etaS) << "  " << exp.name << " / " << predictor.name
                 << " (" << std::setprecision(1) << cellS << "s)";
            std::cerr << line.str() << std::endl;
        }
    }
    return 0;
}

}

int main()
{
    return NES::run();
}
