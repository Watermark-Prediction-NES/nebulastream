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
#include <vector>
#include <SliceStore/DefaultTimeBasedSliceStore.hpp>
#include <SliceStore/Slice.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <SliceCacheConfiguration.hpp>
#include <SlicePreallocationConfiguration.hpp>

namespace NES
{

namespace
{
constexpr uint64_t WindowSize = 100;
constexpr uint64_t WindowSlide = 10;
}

class DefaultTimeBasedSliceStoreTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("DefaultTimeBasedSliceStoreTest.log", LogLevel::LOG_DEBUG); }

    static SlicePreallocationConfiguration makePreallocConfig(const uint64_t horizonMs, const uint64_t poolSize)
    {
        SlicePreallocationConfiguration config{};
        config.preemptiveCreateHorizonMs.setValue(horizonMs);
        config.recyclePoolSize.setValue(poolSize);
        return config;
    }
};

/// Preemptive create: feeding one tuple at ts=50 with horizonMs=20, slide=10 creates the slice
/// for ts=50 plus 2 lookahead slices (sliceEnd + slide, sliceEnd + 2*slide). createNewSlice runs 3×.
TEST_F(DefaultTimeBasedSliceStoreTest, preemptiveCreateMakesNSlicesAhead)
{
    DefaultTimeBasedSliceStore store{WindowSize, WindowSlide, SliceCacheConfiguration{}, makePreallocConfig(/*horizonMs*/ 20, /*pool*/ 0)};
    store.incrementNumberOfInputPipelines();

    uint64_t createCount = 0;
    auto createNewSlice = [&](const SliceStart s, const SliceEnd e) -> std::vector<std::shared_ptr<Slice>>
    {
        ++createCount;
        return {std::make_shared<Slice>(s, e)};
    };

    const auto slices = store.getSlicesOrCreate(Timestamp{50}, createNewSlice);
    ASSERT_EQ(slices.size(), 1u);
    ASSERT_EQ(createCount, 3u) << "expected 1 on-demand + 2 preemptive slices";

    /// The preemptive slices must be findable via getSliceBySliceEnd. Slice ending at 60 is the
    /// on-demand one (ts=50 falls in [50, 60)); lookahead covers [60, 80) → slices ending at 70, 80.
    ASSERT_TRUE(store.getSliceBySliceEnd(SliceEnd{60}).has_value());
    ASSERT_TRUE(store.getSliceBySliceEnd(SliceEnd{70}).has_value());
    ASSERT_TRUE(store.getSliceBySliceEnd(SliceEnd{80}).has_value());
}

/// Recycle round-trip: GC pushes retired slices into the pool; the next on-demand create pops one
/// and calls reset() instead of allocating. shared_ptr identity proves reuse.
TEST_F(DefaultTimeBasedSliceStoreTest, recyclePoolReusesSlicesAcrossGc)
{
    DefaultTimeBasedSliceStore store{WindowSize, WindowSlide, SliceCacheConfiguration{}, makePreallocConfig(/*horizonMs*/ 0, /*pool*/ 4)};
    store.incrementNumberOfInputPipelines();

    uint64_t createCount = 0;
    auto createNewSlice = [&](const SliceStart s, const SliceEnd e) -> std::vector<std::shared_ptr<Slice>>
    {
        ++createCount;
        return {std::make_shared<Slice>(s, e)};
    };

    /// Build one slice [50, 60).
    auto first = store.getSlicesOrCreate(Timestamp{50}, createNewSlice).front();
    auto* firstPtr = first.get();
    ASSERT_EQ(createCount, 1u);

    /// Drain all references the windows / map hold so use_count drops to 1 inside GC.
    first.reset();
    (void)store.getTriggerableWindowSlices(Timestamp{10'000});
    store.garbageCollectSlicesAndWindows(Timestamp{10'000});

    ASSERT_FALSE(store.getSliceBySliceEnd(SliceEnd{60}).has_value()) << "slice should have left the map";

    /// New build at a different timestamp should pop the recycled slice and reset it — no new create.
    auto recycled = store.getSlicesOrCreate(Timestamp{200}, createNewSlice).front();
    EXPECT_EQ(createCount, 1u) << "second build should reuse pooled slice, not allocate";
    EXPECT_EQ(recycled.get(), firstPtr) << "shared_ptr identity proves the same Slice object came back";
    EXPECT_EQ(recycled->getSliceEnd(), SliceEnd{210}) << "reset() should have applied the new sliceEnd";
}

/// When both knobs are off (default config), behaviour matches the unmodified store: exactly one
/// create per timestamp, nothing recycled.
TEST_F(DefaultTimeBasedSliceStoreTest, disabledByDefaultMatchesLegacyBehaviour)
{
    DefaultTimeBasedSliceStore store{WindowSize, WindowSlide, SliceCacheConfiguration{}, SlicePreallocationConfiguration{}};
    store.incrementNumberOfInputPipelines();

    uint64_t createCount = 0;
    auto createNewSlice = [&](const SliceStart s, const SliceEnd e) -> std::vector<std::shared_ptr<Slice>>
    {
        ++createCount;
        return {std::make_shared<Slice>(s, e)};
    };

    (void)store.getSlicesOrCreate(Timestamp{50}, createNewSlice);
    EXPECT_EQ(createCount, 1u);

    (void)store.getTriggerableWindowSlices(Timestamp{10'000});
    store.garbageCollectSlicesAndWindows(Timestamp{10'000});

    (void)store.getSlicesOrCreate(Timestamp{200}, createNewSlice);
    EXPECT_EQ(createCount, 2u) << "with pool disabled, second build must allocate fresh";
}

}
