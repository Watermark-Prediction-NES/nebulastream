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

#include <array>
#include <chrono>
#include <cstdint>
#include <utility>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/ConstantPressureSensor.hpp>
#include <SliceStore/Spill/NeverSpillPolicy.hpp>
#include <SliceStore/Spill/PressureSpillPolicy.hpp>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <SliceStore/Spill/TieredSpillPolicy.hpp>
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

    static SliceSpillContext makeCtx(uint64_t residentBytes, SliceTier currentTier = SliceTier::Resident)
    {
        return SliceSpillContext{
            .sliceEnd = Timestamp{100},
            .now = Timestamp{500},
            .currentTier = currentTier,
            .residentBytes = residentBytes,
            .spilledBytes = 0,
            .windowState = WindowInfoState::WINDOW_FILLING,
        };
    }

    /// Warms a policy's predictor with a slope-1 feed, so predictWallClock(sliceEnd) is exactly
    /// `wallClock + (sliceEnd - watermark)` and a test can name a time-to-trigger directly.
    /// Returns the final (watermark, wallClock) point.
    static std::pair<uint64_t, uint64_t> warmSlopeOne(SpillPolicy& policy, uint64_t ticks = 8)
    {
        /// Deliberately offset the two clocks: if observe() ever swapped its arguments, the resulting
        /// predictions would be off by 4000 and every band assertion below would fail.
        uint64_t watermark = 1000;
        uint64_t wallClock = 5000;
        for (uint64_t i = 0; i < ticks; ++i)
        {
            policy.observe(Timestamp{wallClock}, Timestamp{watermark});
            watermark += 10;
            wallClock += 10;
        }
        return {watermark - 10, wallClock - 10};
    }

    /// A context whose slice triggers exactly `timeToTrigger` ms from now under a slope-1 predictor.
    static SliceSpillContext ctxWithTimeToTrigger(std::pair<uint64_t, uint64_t> endpoint, uint64_t timeToTrigger, SliceTier currentTier)
    {
        auto ctx = makeCtx(1024, currentTier);
        ctx.sliceEnd = SliceEnd{endpoint.first + timeToTrigger};
        ctx.now = Timestamp{endpoint.second};
        return ctx;
    }
};

TEST_F(SpillPolicyContractTest, NeverSpillLeavesEveryTierAloneRegardlessOfPressure)
{
    NeverSpillPolicy policy;
    for (const auto tier : {SliceTier::Resident, SliceTier::CompressedRam, SliceTier::CompressedDisk, SliceTier::Disk})
    {
        for (const double pressure : {0.0, 0.5, 0.95, 1.0})
        {
            EXPECT_EQ(policy.decide(makeCtx(1024, tier), pressure), tier);
        }
    }
}

TEST_F(SpillPolicyContractTest, NeverSpillObserveIsNoop)
{
    NeverSpillPolicy policy;
    /// observe() must not throw and must not change subsequent decisions.
    policy.observe(Timestamp{500}, Timestamp{400});
    EXPECT_EQ(policy.decide(makeCtx(1024), 0.99), SliceTier::Resident);
}

/// The tiered policy is the only one allowed to name a compressed tier. If the pre-existing policies
/// ever start emitting one, a query configured for `reactive` would silently begin compressing.
TEST_F(SpillPolicyContractTest, ExistingPoliciesNeverEmitCompressedTiers)
{
    NeverSpillPolicy never;
    PressureSpillPolicy reactive{0.8};
    PressureSpillPolicy predictive{0.8, std::chrono::milliseconds{50}, "ewma"};
    warmSlopeOne(predictive);
    const std::array<const SpillPolicy*, 3> policies{&never, &reactive, &predictive};

    for (const auto tier : {SliceTier::Resident, SliceTier::CompressedRam, SliceTier::CompressedDisk, SliceTier::Disk})
    {
        for (const double pressure : {0.0, 0.5, 0.79, 0.8, 0.95, 1.0})
        {
            for (const auto* policy : policies)
            {
                const auto decision = policy->decide(makeCtx(1024, tier), pressure);
                EXPECT_TRUE(decision == tier || decision == SliceTier::Disk)
                    << "tier=" << static_cast<uint32_t>(tier) << " pressure=" << pressure
                    << " decision=" << static_cast<uint32_t>(decision);
            }
        }
    }
}

/// A spilled slice must stay spilled when pressure drops. Returning Resident here instead of the
/// current tier would make the store restore every spilled slice the moment memory frees up.
TEST_F(SpillPolicyContractTest, ReactiveKeepsSpilledSlicesSpilledBelowTheBound)
{
    PressureSpillPolicy reactive{0.8};
    EXPECT_EQ(reactive.decide(makeCtx(1024, SliceTier::Disk), 0.1), SliceTier::Disk);
    EXPECT_EQ(reactive.decide(makeCtx(1024, SliceTier::Resident), 0.1), SliceTier::Resident);
}

/// The "always" registry name is PressureSpillPolicy with a zero bound. sample() is clamped to [0, 1],
/// so the `pressure < highBound` gate can never fire and every slice is evicted on every tick.
TEST_F(SpillPolicyContractTest, AlwaysEvictsRegardlessOfPressure)
{
    PressureSpillPolicy always{0.0};
    for (const double pressure : {0.0, 0.01, 0.5, 0.95, 1.0})
    {
        EXPECT_EQ(always.decide(makeCtx(1024, SliceTier::Resident), pressure), SliceTier::Disk) << "pressure=" << pressure;
        /// Already evicted stays evicted — the policy never names Resident, so nothing is ever restored
        /// except by the probe.
        EXPECT_EQ(always.decide(makeCtx(1024, SliceTier::Disk), pressure), SliceTier::Disk) << "pressure=" << pressure;
    }
}

/// The distinction that makes "always" worth its own name: reactive holds slices resident below its
/// bound, always never does.
TEST_F(SpillPolicyContractTest, AlwaysDiffersFromReactiveBelowTheBound)
{
    PressureSpillPolicy reactive{0.8};
    PressureSpillPolicy always{0.0};
    EXPECT_EQ(reactive.decide(makeCtx(1024, SliceTier::Resident), 0.5), SliceTier::Resident);
    EXPECT_EQ(always.decide(makeCtx(1024, SliceTier::Resident), 0.5), SliceTier::Disk);
}

TEST_F(SpillPolicyContractTest, TieredBelowPressureBoundLeavesSliceInPlace)
{
    TieredSpillPolicy policy{
        0.8,
        std::chrono::milliseconds{20},
        std::chrono::milliseconds{50},
        std::chrono::milliseconds{200},
        std::chrono::milliseconds{1000},
        "ewma"};
    const auto endpoint = warmSlopeOne(policy);
    for (const auto tier : {SliceTier::Resident, SliceTier::CompressedRam, SliceTier::Disk})
    {
        /// 5000 ms out — far past every band, but pressure is fine, so nothing moves.
        EXPECT_EQ(policy.decide(ctxWithTimeToTrigger(endpoint, 5000, tier), 0.1), tier);
    }
}

TEST_F(SpillPolicyContractTest, TieredWithColdPredictorMatchesReactive)
{
    TieredSpillPolicy tiered{
        0.8,
        std::chrono::milliseconds{20},
        std::chrono::milliseconds{50},
        std::chrono::milliseconds{200},
        std::chrono::milliseconds{1000},
        "off"};
    PressureSpillPolicy reactive{0.8};
    for (const auto tier : {SliceTier::Resident, SliceTier::CompressedRam, SliceTier::Disk})
    {
        for (const double pressure : {0.0, 0.5, 0.79, 0.8, 0.95, 1.0})
        {
            EXPECT_EQ(tiered.decide(makeCtx(1024, tier), pressure), reactive.decide(makeCtx(1024, tier), pressure))
                << "tier=" << static_cast<uint32_t>(tier) << " pressure=" << pressure;
        }
    }
}

TEST_F(SpillPolicyContractTest, TieredLadderBandsAreDeterministic)
{
    TieredSpillPolicy policy{
        0.8,
        std::chrono::milliseconds{20},
        std::chrono::milliseconds{50},
        std::chrono::milliseconds{200},
        std::chrono::milliseconds{1000},
        "ewma"};
    const auto endpoint = warmSlopeOne(policy);

    /// Each band is closed at its upper edge; one ms past it belongs to the next band down.
    const struct
    {
        uint64_t timeToTrigger;
        SliceTier expected;
    } cases[] = {
        {0, SliceTier::Resident},
        {20, SliceTier::Resident},
        {21, SliceTier::Resident},
        {50, SliceTier::Resident},
        {51, SliceTier::CompressedRam},
        {200, SliceTier::CompressedRam},
        {201, SliceTier::CompressedDisk},
        {1000, SliceTier::CompressedDisk},
        {1001, SliceTier::Disk},
        {100000, SliceTier::Disk},
    };

    for (const auto& c : cases)
    {
        EXPECT_EQ(policy.decide(ctxWithTimeToTrigger(endpoint, c.timeToTrigger, SliceTier::Resident), 0.95), c.expected)
            << "timeToTrigger=" << c.timeToTrigger;
    }
}

TEST_F(SpillPolicyContractTest, TieredPromotesSpilledSliceInsidePromoteHorizonEvenUnderPressure)
{
    TieredSpillPolicy policy{
        0.8,
        std::chrono::milliseconds{20},
        std::chrono::milliseconds{50},
        std::chrono::milliseconds{200},
        std::chrono::milliseconds{1000},
        "ewma"};
    const auto endpoint = warmSlopeOne(policy);

    /// Promotion outranks the pressure gate — the probe is about to arrive, so the slice must be back
    /// even though memory is tight. Note this fires below the bound too, which is what keeps the
    /// ladder monotone: time-to-trigger only shrinks, so a slice never leaves the promote band.
    for (const auto tier : {SliceTier::CompressedRam, SliceTier::CompressedDisk, SliceTier::Disk})
    {
        EXPECT_EQ(policy.decide(ctxWithTimeToTrigger(endpoint, 10, tier), 1.0), SliceTier::Resident);
        EXPECT_EQ(policy.decide(ctxWithTimeToTrigger(endpoint, 10, tier), 0.1), SliceTier::Resident);
    }
}

TEST_F(SpillPolicyContractTest, TieredHorizonsAreClampedMonotone)
{
    /// Horizons given out of order: far < mid < near. Clamping must leave a total, ordered ladder
    /// rather than an unreachable band.
    TieredSpillPolicy policy{
        0.8,
        std::chrono::milliseconds{500},
        std::chrono::milliseconds{400},
        std::chrono::milliseconds{300},
        std::chrono::milliseconds{200},
        "ewma"};
    const auto endpoint = warmSlopeOne(policy);

    /// Everything collapses to promote=near=ram=disk=500, so 500 promotes and 501 spills.
    EXPECT_EQ(policy.decide(ctxWithTimeToTrigger(endpoint, 500, SliceTier::Disk), 0.95), SliceTier::Resident);
    EXPECT_EQ(policy.decide(ctxWithTimeToTrigger(endpoint, 501, SliceTier::Resident), 0.95), SliceTier::Disk);
}

TEST_F(SpillPolicyContractTest, ConstantPressureSensorReturnsSetValue)
{
    ConstantPressureSensor sensor{0.42};
    EXPECT_DOUBLE_EQ(sensor.sample(), 0.42);
    sensor.set(0.99);
    EXPECT_DOUBLE_EQ(sensor.sample(), 0.99);
}

}
