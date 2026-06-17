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
#include <memory>
#include <string>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <Watermark/EwmaWatermarkPredictor.hpp>
#include <Watermark/KalmanWatermarkPredictor.hpp>
#include <Watermark/WatermarkPredictor.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <WatermarkPredictorTestUtils.hpp>

namespace NES
{

/// Parameterized fixture so the two predictors share the same behavioural tests.
class WatermarkPredictorBehaviourTest : public Testing::BaseUnitTest, public ::testing::WithParamInterface<std::string>
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("WatermarkPredictorBehaviourTest.log", LogLevel::LOG_DEBUG); }

    [[nodiscard]] std::unique_ptr<WatermarkPredictor> make() const
    {
        if (GetParam() == "ewma")
        {
            return std::make_unique<EwmaWatermarkPredictor>(0.3);
        }
        return std::make_unique<KalmanWatermarkPredictor>();
    }
};

TEST_P(WatermarkPredictorBehaviourTest, NoObservationsReturnsInvalid)
{
    const auto p = make();
    EXPECT_EQ(p->predictWallClock(T(500)).getRawValue(), INVALID_TS);
}

TEST_P(WatermarkPredictorBehaviourTest, TargetAlreadyReachedReturnsLastWallClock)
{
    auto p = make();
    const auto end = feedConstantRate(*p, 10, 100, 50);
    /// Target below the last seen watermark -> window already fired before the last observation.
    const auto predicted = p->predictWallClock(T(end.lastWatermark - 200));
    /// EWMA: lastWatermark is exact; Kalman: x0 may differ slightly. Predicted must equal lastWallClock.
    EXPECT_EQ(predicted.getRawValue(), end.lastWallClock);
}

TEST_P(WatermarkPredictorBehaviourTest, ConstantRateExtrapolatesAccurately)
{
    /// Rate = 100 event-time units per 50 wall-clock units = 2.0
    auto p = make();
    const auto end = feedConstantRate(*p, 50, 100, 50);
    /// Predict crossing at 2000 event units past the last watermark.
    /// At rate 2.0 -> dt = 1000 wall units -> expected ~ lastWallClock + 1000.
    const auto predicted = p->predictWallClock(T(end.lastWatermark + 2000));
    ASSERT_NE(predicted.getRawValue(), INVALID_TS);
    const uint64_t expected = end.lastWallClock + 1000;
    const uint64_t err = predicted.getRawValue() > expected ? predicted.getRawValue() - expected : expected - predicted.getRawValue();
    EXPECT_LE(err, 50U) << "predicted=" << predicted.getRawValue() << " expected=" << expected;
}

TEST_P(WatermarkPredictorBehaviourTest, DuplicateWallClockIsIgnored)
{
    auto p = make();
    p->observe(T(1000), T(100));
    for (int i = 0; i < 20; ++i)
    {
        p->observe(T(2000 + i), T(100));
    }
    /// Without a second non-duplicate, EWMA still has insufficient state.
    if (GetParam() == "ewma")
    {
        EXPECT_EQ(p->predictWallClock(T(5000)).getRawValue(), INVALID_TS);
    }
    p->observe(T(2000), T(200));
    EXPECT_NE(p->predictWallClock(T(5000)).getRawValue(), INVALID_TS);
}

TEST_P(WatermarkPredictorBehaviourTest, AdaptsAfterRateChange)
{
    auto p = make();
    /// Phase 1: rate 1.0
    const auto endPhase1 = feedConstantRate(*p, 80, 50, 50);
    /// Phase 2: rate 4.0
    const auto endPhase2 = feedConstantRate(*p, 200, 200, 50, endPhase1.lastWatermark + 50, endPhase1.lastWallClock + 50);

    /// 1000 event units past the last watermark should take ~250 wall units at rate 4.0.
    const auto predicted = p->predictWallClock(T(endPhase2.lastWatermark + 1000));
    ASSERT_NE(predicted.getRawValue(), INVALID_TS);
    const uint64_t observedDt = predicted.getRawValue() - endPhase2.lastWallClock;
    EXPECT_GE(observedDt, 200U);
    EXPECT_LE(observedDt, 500U);
}

TEST_P(WatermarkPredictorBehaviourTest, PredictionForLastWatermarkEqualsLastWallClock)
{
    auto p = make();
    const auto end = feedConstantRate(*p, 20, 100, 50);
    /// Asking "when does the watermark reach the value it just had" -> it's already there.
    EXPECT_EQ(p->predictWallClock(T(end.lastWatermark)).getRawValue(), end.lastWallClock);
}

INSTANTIATE_TEST_SUITE_P(
    Predictors, WatermarkPredictorBehaviourTest, ::testing::Values("ewma", "kalman"), [](const auto& info) { return info.param; });

}
