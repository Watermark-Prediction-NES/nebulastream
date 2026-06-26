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
#include <Watermark/KalmanWatermarkPredictor.hpp>
#include <Watermark/RobustAdaptiveKalmanWatermarkPredictor.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <WatermarkPredictorTestUtils.hpp>

namespace NES
{

class RobustAdaptiveKalmanWatermarkPredictorTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("RobustAdaptiveKalmanWatermarkPredictorTest.log", LogLevel::LOG_DEBUG); }
};

TEST_F(RobustAdaptiveKalmanWatermarkPredictorTest, ValidAfterFirstObservation)
{
    /// Initialises on the first sample like the plain Kalman, so it must already return a finite value.
    RobustAdaptiveKalmanWatermarkPredictor p;
    p.observe(T(1000), T(100));
    EXPECT_NE(p.predictWallClock(T(2000)).getRawValue(), INVALID_TS);
}

TEST_F(RobustAdaptiveKalmanWatermarkPredictorTest, ConvergesOnConstantRate)
{
    RobustAdaptiveKalmanWatermarkPredictor p;
    feedConstantRate(p, 200, 100, 50);
    EXPECT_NEAR(p.currentRateEstimate(), 2.0, 0.02);
}

TEST_F(RobustAdaptiveKalmanWatermarkPredictorTest, NonPositiveRateDoesNotProduceInvalid)
{
    /// Flat sequence (rate ~ 0): predictor must clamp the rate and return a finite Timestamp.
    RobustAdaptiveKalmanWatermarkPredictor p;
    for (uint64_t i = 0; i < 50; ++i)
    {
        p.observe(T(1000), T(100 + i));
    }
    EXPECT_NE(p.predictWallClock(T(2000)).getRawValue(), INVALID_TS);
}

TEST_F(RobustAdaptiveKalmanWatermarkPredictorTest, GatesSingleStragglerOutlier)
{
    /// After a clean constant-rate prefix, inject one wildly off watermark. The WoLF gate must keep
    /// the rate estimate near the true rate, whereas the plain Kalman is dragged much further.
    RobustAdaptiveKalmanWatermarkPredictor robust;
    KalmanWatermarkPredictor plain;
    const auto endR = feedConstantRate(robust, 100, 100, 50);
    const auto endP = feedConstantRate(plain, 100, 100, 50);
    ASSERT_EQ(endR.lastWatermark, endP.lastWatermark);

    /// One outlier sample: watermark spikes +5000 over the expected +100 at the normal wall step.
    robust.observe(T(endR.lastWatermark + 5000), T(endR.lastWallClock + 50));
    plain.observe(T(endP.lastWatermark + 5000), T(endP.lastWallClock + 50));

    const double robustDev = std::abs(robust.currentRateEstimate() - 2.0);
    const double plainDev = std::abs(plain.currentRateEstimate() - 2.0);
    EXPECT_LT(robustDev, 0.5) << "robust rate=" << robust.currentRateEstimate();
    EXPECT_GT(plainDev, robustDev) << "plain rate=" << plain.currentRateEstimate();
}

TEST_F(RobustAdaptiveKalmanWatermarkPredictorTest, AdaptsAfterRateChangeViaRegimeBump)
{
    /// Rate 1.0 -> 4.0. The signed-run detector should bump the process noise so the rate re-adapts
    /// toward 4.0 within a modest number of post-change samples.
    RobustAdaptiveKalmanWatermarkPredictor p;
    const auto endPhase1 = feedConstantRate(p, 80, 50, 50);
    feedConstantRate(p, 60, 200, 50, endPhase1.lastWatermark + 50, endPhase1.lastWallClock + 50);
    EXPECT_NEAR(p.currentRateEstimate(), 4.0, 0.5) << "rate=" << p.currentRateEstimate();
}

TEST_F(RobustAdaptiveKalmanWatermarkPredictorTest, AdaptsObservationNoiseUpwardUnderJitter)
{
    /// Symmetric jitter on the watermark should drive the covariance-matched R above its noise-free
    /// floor, while a clean stream keeps it near the floor.
    RobustAdaptiveKalmanWatermarkPredictor clean;
    feedConstantRate(clean, 200, 100, 50);

    RobustAdaptiveKalmanWatermarkPredictor jittery;
    uint64_t w = 1000;
    uint64_t t = 1000;
    for (uint64_t i = 0; i < 200; ++i)
    {
        /// Deterministic +/- 40 zig-zag around the true watermark -> symmetric measurement noise.
        const int64_t wobble = (i % 2 == 0) ? 40 : -40;
        jittery.observe(T(static_cast<uint64_t>(static_cast<int64_t>(w) + wobble)), T(t));
        w += 100;
        t += 50;
    }
    EXPECT_GT(jittery.currentObservationNoise(), clean.currentObservationNoise());
}

TEST_F(RobustAdaptiveKalmanWatermarkPredictorTest, GateDisabledReducesToPlainKalman)
{
    /// With the gate off (wolfThreshold <= 0) and R-adaptation off, the robust filter must track the
    /// plain Kalman exactly on a clean stream (same base recursion).
    RobustAdaptiveKalmanWatermarkPredictor::Config cfg;
    cfg.wolfThreshold = 0.0;
    cfg.rAdaptForgetting = 0.0;
    RobustAdaptiveKalmanWatermarkPredictor robust{cfg};
    KalmanWatermarkPredictor plain;
    feedConstantRate(robust, 100, 100, 50);
    feedConstantRate(plain, 100, 100, 50);
    EXPECT_NEAR(robust.currentRateEstimate(), plain.currentRateEstimate(), 1e-9);
}

}
