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

#include <cstdint>
#include <random>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <Watermark/KalmanWatermarkPredictor.hpp>
#include <Watermark/NeuralKalmanWatermarkPredictor.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <WatermarkPredictorTestUtils.hpp>

namespace NES
{

class NeuralKalmanWatermarkPredictorTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("NeuralKalmanWatermarkPredictorTest.log", LogLevel::LOG_DEBUG); }
};

/// Insufficient state before the first observation.
TEST_F(NeuralKalmanWatermarkPredictorTest, NoObservationsReturnsInvalid)
{
    NeuralKalmanWatermarkPredictor p;
    EXPECT_EQ(p.predictWallClock(T(1000)).getRawValue(), INVALID_TS);
}

/// On a perfectly constant rate the filter converges to the exact rate (the learned gain stays
/// near 1, i.e. it tracks the plain Kalman optimum on clean data).
TEST_F(NeuralKalmanWatermarkPredictorTest, ConvergesToConstantRate)
{
    NeuralKalmanWatermarkPredictor p;
    /// rate = 100 / 50 = 2.0
    const auto end = feedConstantRate(p, 200, 100, 50);
    EXPECT_NEAR(p.currentRateEstimate(), 2.0, 0.05);

    const auto predicted = p.predictWallClock(T(end.lastWatermark + 2000));
    ASSERT_NE(predicted.getRawValue(), INVALID_TS);
    const uint64_t expected = end.lastWallClock + 1000;
    const uint64_t err = predicted.getRawValue() > expected ? predicted.getRawValue() - expected : expected - predicted.getRawValue();
    EXPECT_LE(err, 30U) << "predicted=" << predicted.getRawValue() << " expected=" << expected;
}

/// Graceful init: the gain modulation starts at ~1, so on a clean feed the predictor tracks the
/// plain Kalman filter closely -- it is never worse than that baseline before the net has trained.
TEST_F(NeuralKalmanWatermarkPredictorTest, BehavesLikeKalmanOnCleanFeed)
{
    NeuralKalmanWatermarkPredictor neural;
    KalmanWatermarkPredictor kalman; /// matching default constant-velocity config

    const auto neuralEnd = feedConstantRate(neural, 100, 100, 50);
    const auto kalmanEnd = feedConstantRate(kalman, 100, 100, 50);
    ASSERT_EQ(neuralEnd.lastWatermark, kalmanEnd.lastWatermark);

    for (const uint64_t horizon : {500U, 2000U, 5000U})
    {
        const auto pn = neural.predictWallClock(T(neuralEnd.lastWatermark + horizon));
        const auto pk = kalman.predictWallClock(T(kalmanEnd.lastWatermark + horizon));
        ASSERT_NE(pn.getRawValue(), INVALID_TS);
        ASSERT_NE(pk.getRawValue(), INVALID_TS);
        const uint64_t diff
            = pn.getRawValue() > pk.getRawValue() ? pn.getRawValue() - pk.getRawValue() : pk.getRawValue() - pn.getRawValue();
        EXPECT_LE(diff, 50U) << "horizon=" << horizon << " neural=" << pn.getRawValue() << " kalman=" << pk.getRawValue();
    }
}

/// A sustained rate increase must be tracked.
TEST_F(NeuralKalmanWatermarkPredictorTest, AdaptsToRateIncrease)
{
    NeuralKalmanWatermarkPredictor p;
    /// Phase 1: rate 1.0
    const auto phase1 = feedConstantRate(p, 60, 50, 50);
    /// Phase 2: rate 6.0
    const auto phase2 = feedConstantRate(p, 150, 300, 50, phase1.lastWatermark + 50, phase1.lastWallClock + 50);

    /// 3000 event units past the last watermark should take ~500 wall units at rate 6.0.
    const auto predicted = p.predictWallClock(T(phase2.lastWatermark + 3000));
    ASSERT_NE(predicted.getRawValue(), INVALID_TS);
    const uint64_t observedDt = predicted.getRawValue() - phase2.lastWallClock;
    EXPECT_GE(observedDt, 300U);
    EXPECT_LE(observedDt, 800U);
}

/// A single late-arrival straggler must not derail the estimate: the Huber loss caps its training
/// pull and the analytic-gain covariance keeps the update bounded.
TEST_F(NeuralKalmanWatermarkPredictorTest, RobustToSingleStraggler)
{
    NeuralKalmanWatermarkPredictor p;
    /// 50 clean steps at rate 2.0.
    auto end = feedConstantRate(p, 50, 100, 50);
    /// One straggler: a big wall-clock gap with a small watermark advance.
    p.observe(T(end.lastWatermark + 100), T(end.lastWallClock + 500));
    /// 30 more clean steps at rate 2.0.
    end = feedConstantRate(p, 30, 100, 50, end.lastWatermark + 200, end.lastWallClock + 550);

    EXPECT_NEAR(p.currentRateEstimate(), 2.0, 0.6) << "straggler derailed the rate estimate";
}

/// Zero-mean wall-clock jitter must not derail the rate estimate (the whole point of the filter).
TEST_F(NeuralKalmanWatermarkPredictorTest, JitterDoesNotDerailRate)
{
    NeuralKalmanWatermarkPredictor p;
    std::mt19937_64 rng{7};
    std::normal_distribution<double> jitter{0.0, 10.0};

    uint64_t watermark = 1000;
    uint64_t wall = 1000;
    for (int i = 0; i < 300; ++i)
    {
        p.observe(T(watermark), T(wall));
        watermark += 100; /// true rate 100 / 50 = 2.0
        const auto step = static_cast<int64_t>(50.0 + jitter(rng));
        wall += static_cast<uint64_t>(step > 1 ? step : 1); /// keep wall-clock strictly increasing
    }
    EXPECT_NEAR(p.currentRateEstimate(), 2.0, 0.3) << "jitter derailed the rate estimate";
}

/// Same seed + same input stream -> identical predictions (no hidden nondeterminism).
TEST_F(NeuralKalmanWatermarkPredictorTest, DeterministicForFixedSeed)
{
    const auto drive = [](NeuralKalmanWatermarkPredictor& p)
    {
        const auto phase1 = feedConstantRate(p, 40, 50, 50);
        return feedConstantRate(p, 80, 300, 50, phase1.lastWatermark + 50, phase1.lastWallClock + 50);
    };

    NeuralKalmanWatermarkPredictor a;
    NeuralKalmanWatermarkPredictor b;
    const auto endA = drive(a);
    const auto endB = drive(b);
    ASSERT_EQ(endA.lastWatermark, endB.lastWatermark);

    for (const uint64_t horizon : {500U, 2000U, 5000U})
    {
        const auto pa = a.predictWallClock(T(endA.lastWatermark + horizon));
        const auto pb = b.predictWallClock(T(endB.lastWatermark + horizon));
        EXPECT_EQ(pa.getRawValue(), pb.getRawValue()) << "horizon=" << horizon;
    }
}

/// Predicting a farther target never yields an earlier wall-clock time.
TEST_F(NeuralKalmanWatermarkPredictorTest, PredictionMonotoneInHorizon)
{
    NeuralKalmanWatermarkPredictor p;
    const auto end = feedConstantRate(p, 100, 100, 50);
    const auto near = p.predictWallClock(T(end.lastWatermark + 500));
    const auto mid = p.predictWallClock(T(end.lastWatermark + 2000));
    const auto far = p.predictWallClock(T(end.lastWatermark + 5000));
    ASSERT_NE(near.getRawValue(), INVALID_TS);
    EXPECT_LE(near.getRawValue(), mid.getRawValue());
    EXPECT_LE(mid.getRawValue(), far.getRawValue());
}

}
