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

/// Cooperative slice-group creation: one caller wins a buffer's slice range and creates every missing
/// slice (with full window registration), overlapping callers get a retry delay that always clears, and
/// the group-sizing formula keeps creation ahead of the stream without ever exceeding its cap.

#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>

#include <SliceStore/DefaultTimeBasedSliceStore.hpp>
#include <SliceStore/Slice.hpp>
#include <Time/Timestamp.hpp>
#include <gtest/gtest.h>
#include <SliceCacheConfiguration.hpp>
#include <SliceStoreConfiguration.hpp>

namespace NES
{
namespace
{
constexpr uint64_t WINDOW_SIZE = 100;
constexpr uint64_t WINDOW_SLIDE = 20;

SliceCreateFunction countingCreateFunction(uint64_t* creations)
{
    return [creations](const SliceStart start, const SliceEnd end, const std::shared_ptr<Slice>&) -> std::vector<std::shared_ptr<Slice>>
    {
        ++(*creations);
        return {std::make_shared<Slice>(start, end)};
    };
}
}

TEST(SliceGroupCreationTest, GroupSizeWithoutRateOrCostIsJustTheNeed)
{
    EXPECT_EQ(computeSliceGroupSize(3, 0.0, 1000, WINDOW_SLIDE, 64), 3);
    EXPECT_EQ(computeSliceGroupSize(3, 5.0, 0, WINDOW_SLIDE, 64), 3);
}

TEST(SliceGroupCreationTest, GroupSizeIsCappedWhenTheStreamOutrunsCreation)
{
    /// rate * costMs / slide = 100 * 1ms / 20 = 5 >= 1: creation can never catch up.
    EXPECT_EQ(computeSliceGroupSize(3, 100.0, 1'000'000, WINDOW_SLIDE, 64), 64);
    /// The need alone already exceeds the cap.
    EXPECT_EQ(computeSliceGroupSize(100, 0.0, 0, WINDOW_SLIDE, 64), 64);
}

TEST(SliceGroupCreationTest, GroupSizeExtendsAheadProportionallyToTheRate)
{
    /// f = 10 * 1ms / 20 = 0.5 -> total = ceil(4 / 0.5) = 8.
    EXPECT_EQ(computeSliceGroupSize(4, 10.0, 1'000'000, WINDOW_SLIDE, 64), 8);
}

TEST(SliceGroupCreationTest, WinnerCreatesEverySliceOfTheRange)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, SliceStoreConfiguration{}};
    uint64_t creations = 0;

    const auto delay = store.claimOrDeferSliceRange(Timestamp(5), Timestamp(95), 0.0, countingCreateFunction(&creations));

    EXPECT_EQ(delay, 0U);
    EXPECT_EQ(creations, 5U);
    for (uint64_t sliceEnd = WINDOW_SLIDE; sliceEnd <= WINDOW_SIZE; sliceEnd += WINDOW_SLIDE)
    {
        EXPECT_TRUE(store.getSliceBySliceEnd(Timestamp(sliceEnd)).has_value()) << "missing slice end " << sliceEnd;
    }
}

TEST(SliceGroupCreationTest, CoveredRangeNeedsNoClaimAndNoCreation)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, SliceStoreConfiguration{}};
    uint64_t creations = 0;
    std::ignore = store.claimOrDeferSliceRange(Timestamp(5), Timestamp(95), 0.0, countingCreateFunction(&creations));

    creations = 0;
    const auto delay = store.claimOrDeferSliceRange(Timestamp(0), Timestamp(99), 0.0, countingCreateFunction(&creations));

    EXPECT_EQ(delay, 0U);
    EXPECT_EQ(creations, 0U);
}

TEST(SliceGroupCreationTest, OverlappingClaimDefersUntilTheWinnerFinishes)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, SliceStoreConfiguration{}};

    std::promise<void> winnerStartedCreating;
    std::promise<void> releaseWinner;
    auto winnerBlocked = releaseWinner.get_future().share();
    bool signalled = false;
    const SliceCreateFunction blockingCreate
        = [&](const SliceStart start, const SliceEnd end, const std::shared_ptr<Slice>&) -> std::vector<std::shared_ptr<Slice>>
    {
        if (not signalled)
        {
            signalled = true;
            winnerStartedCreating.set_value();
        }
        winnerBlocked.wait();
        return {std::make_shared<Slice>(start, end)};
    };

    std::thread winner{[&] { std::ignore = store.claimOrDeferSliceRange(Timestamp(5), Timestamp(95), 0.0, blockingCreate); }};
    winnerStartedCreating.get_future().wait();

    /// The winner holds the claim for [5, 95]; an overlapping caller must be told to retry, not create.
    uint64_t loserCreations = 0;
    const auto delay = store.claimOrDeferSliceRange(Timestamp(50), Timestamp(150), 0.0, countingCreateFunction(&loserCreations));
    EXPECT_GE(delay, 1U);
    EXPECT_EQ(loserCreations, 0U);

    releaseWinner.set_value();
    winner.join();

    /// The claim is gone; the same range now completes (only the slices beyond 100 are still missing).
    const auto secondTry = store.claimOrDeferSliceRange(Timestamp(50), Timestamp(150), 0.0, countingCreateFunction(&loserCreations));
    EXPECT_EQ(secondTry, 0U);
    EXPECT_GT(loserCreations, 0U);
}
}
