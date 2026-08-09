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

/// The recycling contract of DefaultTimeBasedSliceStore: garbage collection retires slices into a bounded
/// pool instead of destroying them, retired slices are offered back to the create function which may only
/// reuse a structural match, and every retirement announces the dying slice end exactly once.

#include <cstdint>
#include <memory>
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

/// A slice whose reset succeeds and is observable.
class ResettableSlice final : public Slice
{
public:
    ResettableSlice(const SliceStart start, const SliceEnd end) : Slice(start, end) { }

    [[nodiscard]] bool resetForReuse() override
    {
        wasReset = true;
        return true;
    }

    bool wasReset = false;
};

/// A slice that refuses to be reset, like NLJSlice after a destructive merge.
class NonResettableSlice final : public Slice
{
public:
    NonResettableSlice(const SliceStart start, const SliceEnd end) : Slice(start, end) { }
};

SliceStoreConfiguration makeConfig(const bool recyclingEnabled)
{
    SliceStoreConfiguration config;
    if (recyclingEnabled)
    {
        config.enableSliceRecycling.setValue(true);
    }
    return config;
}

/// offeredCandidates is an optional out-parameter: callers that do not inspect the offers pass nothing, which a reference cannot express.
template <typename SliceType>
/// NOLINTNEXTLINE(fuchsia-default-arguments-declarations)
SliceCreateFunction makeCreateFunction(std::vector<std::shared_ptr<Slice>>* offeredCandidates = nullptr, const bool reuse = true)
{
    return [offeredCandidates, reuse](
               const SliceStart start, const SliceEnd end, const std::shared_ptr<Slice>& candidate) -> std::vector<std::shared_ptr<Slice>>
    {
        if (offeredCandidates != nullptr and candidate != nullptr)
        {
            offeredCandidates->push_back(candidate);
        }
        if (reuse and candidate != nullptr)
        {
            candidate->reassign(start, end);
            return {candidate};
        }
        return {std::make_shared<SliceType>(start, end)};
    };
}

/// Fills one slice per distinct slice range in [0, count) and garbage-collects all of them.
template <typename SliceType>
void createAndRetireSlices(DefaultTimeBasedSliceStore& store, const uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i)
    {
        std::ignore = store.getSlicesOrCreate(Timestamp(i * WINDOW_SLIDE), makeCreateFunction<SliceType>());
    }
    /// All windows containing these slices must have been triggered before garbage collection may retire them.
    std::ignore = store.getTriggerableWindowSlices(Timestamp((count * WINDOW_SLIDE) + (2 * WINDOW_SIZE)));
    store.garbageCollectSlicesAndWindows(Timestamp((count * WINDOW_SLIDE) + (2 * WINDOW_SIZE)));
}
}

TEST(SliceRecyclingTest, GarbageCollectionOffersRetiredSlicesForReuse)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, makeConfig(true)};
    createAndRetireSlices<ResettableSlice>(store, 3);

    std::vector<std::shared_ptr<Slice>> offered;
    const auto slices = store.getSlicesOrCreate(Timestamp(1000), makeCreateFunction<ResettableSlice>(&offered));
    ASSERT_EQ(slices.size(), 1U);
    ASSERT_EQ(offered.size(), 1U);
    EXPECT_EQ(slices[0], offered[0]) << "the reused candidate must be the slice the store returns";
    EXPECT_TRUE(dynamic_cast<ResettableSlice&>(*slices[0]).wasReset);
    EXPECT_EQ(slices[0]->getSliceStart(), Timestamp(1000));
    EXPECT_EQ(slices[0]->getSliceEnd(), Timestamp(1020));
}

TEST(SliceRecyclingTest, RejectedCandidatesAreDroppedNotRestocked)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, makeConfig(true)};
    createAndRetireSlices<ResettableSlice>(store, 1);

    std::vector<std::shared_ptr<Slice>> offered;
    /// The create function refuses the candidate; the store must fall back to the freshly created slice.
    const auto slices = store.getSlicesOrCreate(Timestamp(1000), makeCreateFunction<ResettableSlice>(&offered, false));
    ASSERT_EQ(offered.size(), 1U);
    EXPECT_NE(slices[0], offered[0]);

    /// The pool is now empty: the next miss gets no candidate.
    offered.clear();
    std::ignore = store.getSlicesOrCreate(Timestamp(2000), makeCreateFunction<ResettableSlice>(&offered));
    EXPECT_TRUE(offered.empty());
}

TEST(SliceRecyclingTest, NonResettableSlicesAreDestroyedAtGarbageCollection)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, makeConfig(true)};
    createAndRetireSlices<NonResettableSlice>(store, 3);

    std::vector<std::shared_ptr<Slice>> offered;
    std::ignore = store.getSlicesOrCreate(Timestamp(1000), makeCreateFunction<NonResettableSlice>(&offered));
    EXPECT_TRUE(offered.empty()) << "slices that cannot reset must not reach the pool via garbage collection";
}

TEST(SliceRecyclingTest, RecyclingOffNeverPools)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, makeConfig(false)};
    createAndRetireSlices<ResettableSlice>(store, 3);

    std::vector<std::shared_ptr<Slice>> offered;
    std::ignore = store.getSlicesOrCreate(Timestamp(1000), makeCreateFunction<ResettableSlice>(&offered));
    EXPECT_TRUE(offered.empty());
}

TEST(SliceRecyclingTest, PoolIsCappedAtSteadyStateSliceCount)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, makeConfig(true)};
    /// windowSize / windowSlide = 5 is the cap; retire twice as many slices.
    createAndRetireSlices<ResettableSlice>(store, 10);

    uint64_t candidatesSeen = 0;
    for (uint64_t i = 0; i < 10; ++i)
    {
        std::vector<std::shared_ptr<Slice>> offered;
        std::ignore = store.getSlicesOrCreate(Timestamp(10000 + (i * WINDOW_SLIDE)), makeCreateFunction<ResettableSlice>(&offered, false));
        candidatesSeen += offered.size();
    }
    EXPECT_EQ(candidatesSeen, WINDOW_SIZE / WINDOW_SLIDE);
}

TEST(SliceRecyclingTest, EveryRetiredSliceAnnouncesItsOldSliceEndOnce)
{
    DefaultTimeBasedSliceStore store{WINDOW_SIZE, WINDOW_SLIDE, SliceCacheConfiguration{}, makeConfig(true)};
    std::vector<SliceEnd> retiredEnds;
    store.setOnSliceRetired([&retiredEnds](const SliceEnd end) { retiredEnds.push_back(end); });

    createAndRetireSlices<ResettableSlice>(store, 3);

    EXPECT_EQ(retiredEnds.size(), 3U);
    for (uint64_t i = 0; i < retiredEnds.size(); ++i)
    {
        EXPECT_EQ(retiredEnds[i], Timestamp((i + 1) * WINDOW_SLIDE));
    }
}
}
