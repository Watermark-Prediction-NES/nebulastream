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
#include <vector>
#include <Runtime/BufferManager.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/BufferPoolPressureSensor.hpp>
#include <SliceStore/Spill/ConstantPressureSensor.hpp>
#include <SliceStore/Spill/NeverSpillPolicy.hpp>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>

namespace NES
{

class SpillPolicyContractTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("SpillPolicyContractTest.log", LogLevel::LOG_DEBUG); }

    static SliceSpillContext makeCtx(uint64_t residentBytes)
    {
        return SliceSpillContext{
            .sliceEnd = Timestamp{100},
            .now = Timestamp{500},
            .predictedTriggerWallClock = Timestamp{Timestamp::INVALID_VALUE},
            .residentBytes = residentBytes,
            .spilledBytes = 0,
            .windowState = WindowInfoState::WINDOW_FILLING,
        };
    }
};

TEST_F(SpillPolicyContractTest, NeverSpillIsAlwaysKeepRegardlessOfPressure)
{
    NeverSpillPolicy policy;
    for (const double pressure : {0.0, 0.5, 0.95, 1.0})
    {
        EXPECT_EQ(policy.decide(makeCtx(1024), pressure), SpillDecision::Keep);
    }
}

TEST_F(SpillPolicyContractTest, NeverSpillObserveIsNoop)
{
    NeverSpillPolicy policy;
    /// observe() must not throw and must not change subsequent decisions.
    policy.observe(Timestamp{500}, Timestamp{400});
    EXPECT_EQ(policy.decide(makeCtx(1024), 0.99), SpillDecision::Keep);
}

TEST_F(SpillPolicyContractTest, ConstantPressureSensorReturnsSetValue)
{
    ConstantPressureSensor sensor{0.42};
    EXPECT_DOUBLE_EQ(sensor.sample(), 0.42);
    sensor.set(0.99);
    EXPECT_DOUBLE_EQ(sensor.sample(), 0.99);
}

/// Regression: the sensor used to divide the *total* pool size by the buffer size in *bytes*, so it
/// reported a constant 0.0 and no slice was ever spilled. Pressure must track pool occupancy.
TEST_F(SpillPolicyContractTest, BufferPoolPressureRisesAsPooledBuffersAreConsumed)
{
    constexpr uint64_t NumOfBuffers = 16;
    auto bufferManager = BufferManager::create(/*bufferSize*/ 4096, NumOfBuffers);
    const BufferPoolPressureSensor sensor{*bufferManager};

    EXPECT_DOUBLE_EQ(sensor.sample(), 0.0);

    std::vector<TupleBuffer> held;
    for (uint64_t i = 0; i < NumOfBuffers / 2; ++i)
    {
        held.push_back(bufferManager->getBufferBlocking());
    }
    EXPECT_DOUBLE_EQ(sensor.sample(), 0.5);

    while (held.size() < NumOfBuffers)
    {
        held.push_back(bufferManager->getBufferBlocking());
    }
    EXPECT_DOUBLE_EQ(sensor.sample(), 1.0);

    /// Returning the buffers must relieve the pressure again.
    held.clear();
    EXPECT_DOUBLE_EQ(sensor.sample(), 0.0);
}

}
