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

/// Per-cell predict() timing aggregated over Repetitions reps.
struct TimingStats
{
    std::size_t reps{0};
    double meanMs{0.0};
    double minMs{0.0};
    double nsPerPredict{0.0};
};

/// Times predictWallClock() only: the predictor is driven through the same observed prefix the
/// accuracy run uses (untimed), then a fixed batch of queries is timed. Deterministic accuracy is
/// unaffected; this just measures prediction cost.
TimingStats timePredictor(const PredictorEntry& entry, const Experiment& exp)
{
    TimingStats stats{.reps = Repetitions, .meanMs = 0.0, .minMs = 0.0, .nsPerPredict = 0.0};
    if (exp.observePrefix == 0 || exp.observePrefix > exp.observed.size())
    {
        return stats;
    }

    std::uint64_t sink = 0;
    double sumMs = 0.0;
    double minMs = std::numeric_limits<double>::max();
    for (std::size_t rep = 0; rep < Repetitions; ++rep)
    {
        auto predictor = entry.make();
        for (std::size_t i = 0; i < exp.observePrefix; ++i)
        {
            predictor->observe(exp.observed[i].watermarkTs, exp.observed[i].wallClock);
        }
        const std::uint64_t base = exp.observed[exp.observePrefix - 1].watermarkTs.getRawValue() + 1;

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
    stats.nsPerPredict = (stats.meanMs * 1e6) / static_cast<double>(PredictQueriesPerRep);
    return stats;
}

std::vector<Experiment> buildExperiments()
{
    const PiecewiseConstantTraceSource constantFast{{{200, 100, 50}}};
    const PiecewiseConstantTraceSource constantSlow{{{200, 25, 50}}};
    const PiecewiseConstantTraceSource rateChange{{{100, 50, 50}, {300, 200, 50}}};
    const PiecewiseConstantTraceSource slowdown{{{100, 200, 50}, {300, 100, 50}}};
    const PiecewiseConstantTraceSource threePhase{{{80, 50, 50}, {80, 200, 50}, {120, 100, 50}}};

    std::vector<Experiment> experiments;

    /// Clean baselines.
    experiments.push_back(makeCleanExperiment("ConstantRate(2.0) clean", constantFast, 100, {500, 1000, 2000, 5000}));
    experiments.push_back(makeCleanExperiment("ConstantRate(0.5) clean", constantSlow, 100, {500, 1000, 2000}));

    /// Adaptation-lag sweep on the rate-change trace.
    for (const auto samplesAfter : {2, 5, 10, 20, 50, 100})
    {
        experiments.push_back(makeCleanExperiment(
            "RateChange(1->4) +" + std::to_string(samplesAfter) + " samples clean",
            rateChange,
            100 + static_cast<size_t>(samplesAfter),
            {500, 1000, 2000}));
    }

    /// Mild Gaussian jitter, no extra-late spikes.
    const GaussianNoiseModel mildJitter{{.wallClockStddev = 10.0, .seed = 1}};
    experiments.push_back(makeNoisyExperiment("ConstantRate(2.0) mild jitter (sd=10)", constantFast, mildJitter, 100, {500, 1000, 2000}));
    experiments.push_back(makeNoisyExperiment("RateChange(1->4) +10 + mild jitter", rateChange, mildJitter, 110, {500, 1000, 2000}));

    /// Heavier Gaussian jitter.
    const GaussianNoiseModel heavyJitter{{.wallClockStddev = 30.0, .seed = 2}};
    experiments.push_back(makeNoisyExperiment("ConstantRate(2.0) heavy jitter (sd=30)", constantFast, heavyJitter, 100, {500, 1000, 2000}));
    experiments.push_back(makeNoisyExperiment("RateChange(1->4) +20 + heavy jitter", rateChange, heavyJitter, 120, {500, 1000, 2000}));

    /// Stragglers: spike distribution fixed (mean=400, sd=100); sweep over late-fraction.
    const GaussianNoiseModel stragglersMild{
        {.wallClockStddev = 5.0, .lateProbability = 0.10, .lateExtraDelayMean = 200.0, .lateExtraDelayStddev = 50.0, .seed = 3}};
    experiments.push_back(makeNoisyExperiment("ConstantRate(2.0) + 10% stragglers", constantFast, stragglersMild, 100, {500, 1000, 2000}));
    experiments.push_back(makeNoisyExperiment("RateChange(1->4) +20 + 10% stragglers", rateChange, stragglersMild, 120, {500, 1000, 2000}));

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
            makeNoisyExperiment("ConstantRate(2.0) + " + pctLabel + "% heavy stragglers", constantFast, sweep, 100, {500, 1000, 2000}));
    }

    /// Slowdown and three-phase clean.
    for (const auto samplesAfter : {2, 10, 50})
    {
        experiments.push_back(makeCleanExperiment(
            "Slowdown(4->2) +" + std::to_string(samplesAfter) + " samples clean",
            slowdown,
            100 + static_cast<size_t>(samplesAfter),
            {200, 500, 1000}));
    }
    experiments.push_back(makeCleanExperiment("Three phases (1->4->2) clean", threePhase, 170, {500, 1000, 2000}));

    return experiments;
}

std::vector<PredictorEntry> buildPredictors()
{
    return {
        {.name = "EWMA(alpha=0.3)", .make = [] { return std::make_unique<EwmaWatermarkPredictor>(0.3); }},
        {.name = "EWMA(alpha=1.0)", .make = [] { return std::make_unique<EwmaWatermarkPredictor>(1.0); }},
        {.name = "Kalman(default)", .make = [] { return std::make_unique<KalmanWatermarkPredictor>(); }},
    };
}

void printHeader()
{
    std::cout << std::left << std::setw(46) << "Trace" << std::setw(20) << "Predictor" << std::right << std::setw(10) << "Samples"
              << std::setw(14) << "MAE" << std::setw(14) << "RMSE" << std::setw(12) << "MAPE(%)" << std::setw(14) << "MaxErr"
              << std::setw(8) << "Reps" << std::setw(14) << "MeanMs" << std::setw(14) << "MinMs" << std::setw(14) << "ns/pred" << '\n';
    std::cout << std::string(180, '-') << '\n';
}

void printRow(const std::string& traceName, const std::string& predictorName, const PredictionMetrics& m, const TimingStats& t)
{
    std::cout << std::left << std::setw(46) << traceName << std::setw(20) << predictorName << std::right << std::setw(10) << m.samples
              << std::fixed << std::setprecision(2) << std::setw(14) << m.mae << std::setw(14) << m.rmse << std::setw(12) << m.mape
              << std::setw(14) << m.maxError << std::setw(8) << t.reps << std::setw(14) << t.meanMs << std::setw(14) << t.minMs
              << std::setw(14) << t.nsPerPredict << '\n';
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

    /// Progress goes to stderr so it stays out of the stdout table the Python wrapper parses.
    std::cerr << "[bench] " << totalCells << " cells (" << experiments.size() << " traces x " << predictors.size() << " predictors), "
              << Repetitions << " timed reps each\n";

    printHeader();
    const auto runStart = std::chrono::steady_clock::now();
    std::size_t cell = 0;
    for (const auto& exp : experiments)
    {
        for (const auto& predictor : predictors)
        {
            const auto cellStart = std::chrono::steady_clock::now();
            auto p = predictor.make();
            const auto metrics = runBenchmark(*p, exp.observed, exp.truth, exp.observePrefix, exp.horizons);
            const auto timing = timePredictor(predictor, exp);
            printRow(exp.name, predictor.name, metrics, timing);
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
        std::cout << '\n';
    }
    return 0;
}

}

int main()
{
    return NES::run();
}
