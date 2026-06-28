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

#include <cmath>
#include <cstdint>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <Watermark/MlpWatermarkPredictor.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <WatermarkPredictorTestUtils.hpp>

namespace NES
{

class MlpWatermarkPredictorTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("MlpWatermarkPredictorTest.log", LogLevel::LOG_DEBUG); }
};

/// No rate samples yet -> insufficient state.
TEST_F(MlpWatermarkPredictorTest, NoObservationsReturnsInvalid)
{
    MlpWatermarkPredictor p;
    EXPECT_EQ(p.predictWallClock(T(1000)).getRawValue(), INVALID_TS);
    /// A single observation still has no rate (needs two points for a dt).
    p.observe(T(1000), T(1000));
    EXPECT_EQ(p.predictWallClock(T(1000)).getRawValue(), INVALID_TS);
}

/// On a perfectly constant rate the running mean alone pins the forecast (the normalized
/// features collapse to zero), so the estimate is exact.
TEST_F(MlpWatermarkPredictorTest, ConvergesToConstantRate)
{
    MlpWatermarkPredictor p;
    /// rate = 100 / 50 = 2.0
    const auto end = feedConstantRate(p, 200, 100, 50);
    EXPECT_NEAR(p.currentRateEstimate(), 2.0, 0.05);

    const auto predicted = p.predictWallClock(T(end.lastWatermark + 2000));
    ASSERT_NE(predicted.getRawValue(), INVALID_TS);
    const uint64_t expected = end.lastWallClock + 1000;
    const uint64_t err = predicted.getRawValue() > expected ? predicted.getRawValue() - expected : expected - predicted.getRawValue();
    EXPECT_LE(err, 30U) << "predicted=" << predicted.getRawValue() << " expected=" << expected;
}

/// Before the warmup boundary the predictor falls back to its internal EWMA rate so it is
/// never worse than the EWMA baseline during cold start.
TEST_F(MlpWatermarkPredictorTest, WarmupFallbackIsUsableBeforeTrained)
{
    MlpWatermarkPredictor::Config cfg;
    cfg.warmup = 8;
    MlpWatermarkPredictor p{cfg};
    /// 4 rate samples (< warmup) at rate 2.0.
    const auto end = feedConstantRate(p, 5, 100, 50);
    const auto predicted = p.predictWallClock(T(end.lastWatermark + 1000));
    ASSERT_NE(predicted.getRawValue(), INVALID_TS);
    const uint64_t expected = end.lastWallClock + 500;
    const uint64_t err = predicted.getRawValue() > expected ? predicted.getRawValue() - expected : expected - predicted.getRawValue();
    EXPECT_LE(err, 30U);
}

/// A sustained rate increase must be tracked (the running mean adapts and the net learns the
/// residual on top).
TEST_F(MlpWatermarkPredictorTest, AdaptsToRateIncrease)
{
    MlpWatermarkPredictor p;
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

/// A single late-arrival straggler (one low-rate step) must not derail the estimate: the
/// Huber loss caps its training pull and the window forgets it within windowSize steps.
TEST_F(MlpWatermarkPredictorTest, RobustToSingleStraggler)
{
    MlpWatermarkPredictor p;
    /// 50 clean steps at rate 2.0.
    auto end = feedConstantRate(p, 50, 100, 50);
    /// One straggler: a big wall-clock gap with a small watermark advance -> instantaneous rate ~0.2.
    p.observe(T(end.lastWatermark + 100), T(end.lastWallClock + 500));
    /// 30 more clean steps at rate 2.0.
    end = feedConstantRate(p, 30, 100, 50, end.lastWatermark + 200, end.lastWallClock + 550);

    EXPECT_NEAR(p.currentRateEstimate(), 2.0, 0.6) << "straggler derailed the rate estimate";
}

/// Same seed + same input stream -> identical predictions (no hidden nondeterminism).
TEST_F(MlpWatermarkPredictorTest, DeterministicForFixedSeed)
{
    const auto drive = [](MlpWatermarkPredictor& p)
    {
        const auto phase1 = feedConstantRate(p, 40, 50, 50);
        return feedConstantRate(p, 80, 300, 50, phase1.lastWatermark + 50, phase1.lastWallClock + 50);
    };

    MlpWatermarkPredictor a;
    MlpWatermarkPredictor b;
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
TEST_F(MlpWatermarkPredictorTest, PredictionMonotoneInHorizon)
{
    MlpWatermarkPredictor p;
    const auto end = feedConstantRate(p, 100, 100, 50);
    const auto near = p.predictWallClock(T(end.lastWatermark + 500));
    const auto mid = p.predictWallClock(T(end.lastWatermark + 2000));
    const auto far = p.predictWallClock(T(end.lastWatermark + 5000));
    ASSERT_NE(near.getRawValue(), INVALID_TS);
    EXPECT_LE(near.getRawValue(), mid.getRawValue());
    EXPECT_LE(mid.getRawValue(), far.getRawValue());
}

}
