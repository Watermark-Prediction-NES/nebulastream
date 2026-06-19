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

/// Predictive preemptive create: the predictor converts a WALL-CLOCK horizon into a slice count.
/// We warm it to rate = 10 event-time-ms per wall-clock-ms (watermark 100->200 over wall 10->20), so
/// at now=20 with a 5 ms wall-clock horizon (deadline 25) predictWallClock(end) = 20 + (end-200)/10.
/// Slice ends {220,230,240,250} arrive at {22,23,24,25} <= 25; end 260 arrives at 26 > 25.
TEST_F(DefaultTimeBasedSliceStoreTest, predictivePreemptiveCreateUsesPredictedRate)
{
    auto config = makePreallocConfig(/*horizonMs*/ 5, /*pool*/ 0);
    config.preemptivePredictor.setValue("ewma");
    DefaultTimeBasedSliceStore store{WindowSize, WindowSlide, SliceCacheConfiguration{}, config};
    store.incrementNumberOfInputPipelines();

    Timestamp fakeNow{0};
    store.setWallClockSourceForTesting([&] { return fakeNow; });

    uint64_t createCount = 0;
    auto createNewSlice = [&](const SliceStart s, const SliceEnd e) -> std::vector<std::shared_ptr<Slice>>
    {
        ++createCount;
        return {std::make_shared<Slice>(s, e)};
    };

    /// GC ticks are the predictor's sampling point — feed two observations to establish the rate.
    fakeNow = Timestamp{10};
    store.garbageCollectSlicesAndWindows(Timestamp{100});
    fakeNow = Timestamp{20};
    store.garbageCollectSlicesAndWindows(Timestamp{200});

    const auto slices = store.getSlicesOrCreate(Timestamp{205}, createNewSlice);
    ASSERT_EQ(slices.size(), 1u);
    EXPECT_EQ(createCount, 5u) << "1 on-demand (end 210) + 4 predicted preemptive (ends 220..250)";

    EXPECT_TRUE(store.getSliceBySliceEnd(SliceEnd{210}).has_value());
    EXPECT_TRUE(store.getSliceBySliceEnd(SliceEnd{250}).has_value());
    EXPECT_FALSE(store.getSliceBySliceEnd(SliceEnd{260}).has_value()) << "end 260 falls past the wall-clock horizon";
}

/// preemptive_max_slices caps the predicted burst: same warm-up as above, but the cap of 2 stops
/// creation at ends {220,230} even though the predictor would otherwise reach 250.
TEST_F(DefaultTimeBasedSliceStoreTest, predictivePreemptiveCreateRespectsMaxSlicesCap)
{
    auto config = makePreallocConfig(/*horizonMs*/ 5, /*pool*/ 0);
    config.preemptivePredictor.setValue("ewma");
    config.preemptiveMaxSlices.setValue(2);
    DefaultTimeBasedSliceStore store{WindowSize, WindowSlide, SliceCacheConfiguration{}, config};
    store.incrementNumberOfInputPipelines();

    Timestamp fakeNow{0};
    store.setWallClockSourceForTesting([&] { return fakeNow; });

    uint64_t createCount = 0;
    auto createNewSlice = [&](const SliceStart s, const SliceEnd e) -> std::vector<std::shared_ptr<Slice>>
    {
        ++createCount;
        return {std::make_shared<Slice>(s, e)};
    };

    fakeNow = Timestamp{10};
    store.garbageCollectSlicesAndWindows(Timestamp{100});
    fakeNow = Timestamp{20};
    store.garbageCollectSlicesAndWindows(Timestamp{200});

    (void)store.getSlicesOrCreate(Timestamp{205}, createNewSlice);
    EXPECT_EQ(createCount, 3u) << "1 on-demand + 2 preemptive (capped)";
    EXPECT_TRUE(store.getSliceBySliceEnd(SliceEnd{230}).has_value());
    EXPECT_FALSE(store.getSliceBySliceEnd(SliceEnd{240}).has_value()) << "cap stops creation before end 240";
}

/// Cold predictor (never warmed) cannot answer, so the create path falls back to the static
/// event-time lookahead: horizonMs/slide = 20/10 = 2 slices ahead — identical to the no-predictor case.
TEST_F(DefaultTimeBasedSliceStoreTest, predictiveColdFallsBackToStaticLookahead)
{
    auto config = makePreallocConfig(/*horizonMs*/ 20, /*pool*/ 0);
    config.preemptivePredictor.setValue("ewma");
    DefaultTimeBasedSliceStore store{WindowSize, WindowSlide, SliceCacheConfiguration{}, config};
    store.incrementNumberOfInputPipelines();

    uint64_t createCount = 0;
    auto createNewSlice = [&](const SliceStart s, const SliceEnd e) -> std::vector<std::shared_ptr<Slice>>
    {
        ++createCount;
        return {std::make_shared<Slice>(s, e)};
    };

    (void)store.getSlicesOrCreate(Timestamp{50}, createNewSlice);
    EXPECT_EQ(createCount, 3u) << "cold predictor falls back to static 2-slice lookahead (+1 on-demand)";
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
