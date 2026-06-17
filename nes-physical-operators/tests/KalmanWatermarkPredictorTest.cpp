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
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <Watermark/KalmanWatermarkPredictor.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <WatermarkPredictorTestUtils.hpp>

namespace NES
{

class KalmanWatermarkPredictorTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("KalmanWatermarkPredictorTest.log", LogLevel::LOG_DEBUG); }
};

TEST_F(KalmanWatermarkPredictorTest, ValidAfterFirstObservation)
{
    /// Unlike EWMA the Kalman filter initialises on the first sample, so even before
    /// the second observation it has some state and must return a finite value.
    KalmanWatermarkPredictor p;
    p.observe(T(1000), T(100));
    EXPECT_NE(p.predictWallClock(T(2000)).getRawValue(), INVALID_TS);
}

TEST_F(KalmanWatermarkPredictorTest, ConvergesOnConstantRate)
{
    KalmanWatermarkPredictor p;
    feedConstantRate(p, 200, 100, 50);
    EXPECT_NEAR(p.currentRateEstimate(), 2.0, 0.02);
}

TEST_F(KalmanWatermarkPredictorTest, NonPositiveRateDoesNotProduceInvalid)
{
    /// Force a flat sequence (rate ~ 0). Predictor must clamp rate and return a finite Timestamp.
    KalmanWatermarkPredictor p;
    for (uint64_t i = 0; i < 50; ++i)
    {
        p.observe(T(1000), T(100 + i));
    }
    const auto predicted = p.predictWallClock(T(2000));
    EXPECT_NE(predicted.getRawValue(), INVALID_TS);
}

}
