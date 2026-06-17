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

#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <Watermark/EwmaWatermarkPredictor.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <WatermarkPredictorTestUtils.hpp>

namespace NES
{

class EwmaWatermarkPredictorTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("EwmaWatermarkPredictorTest.log", LogLevel::LOG_DEBUG); }
};

TEST_F(EwmaWatermarkPredictorTest, NeedsAtLeastTwoObservations)
{
    EwmaWatermarkPredictor p{0.3};
    p.observe(T(1000), T(100));
    EXPECT_EQ(p.predictWallClock(T(2000)).getRawValue(), INVALID_TS);
    p.observe(T(1100), T(150));
    EXPECT_NE(p.predictWallClock(T(2000)).getRawValue(), INVALID_TS);
}

TEST_F(EwmaWatermarkPredictorTest, AlphaOneTracksMostRecentRate)
{
    EwmaWatermarkPredictor p{1.0};
    p.observe(T(1000), T(100));
    p.observe(T(1100), T(150)); ///< rate sample = 2.0
    p.observe(T(1500), T(200)); ///< rate sample = 8.0; alpha=1 -> predictor adopts this exactly

    /// Predict 800 event units ahead from (1500, 200) at rate 8.0 -> dt = 100, predicted = 300.
    const auto predicted = p.predictWallClock(T(2300));
    EXPECT_EQ(predicted.getRawValue(), 300U);
}

}
