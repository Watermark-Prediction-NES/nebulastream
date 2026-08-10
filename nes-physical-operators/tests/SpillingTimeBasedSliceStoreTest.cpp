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
#include <expected>
#include <future>
#include <memory>
#include <typeindex>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Interface/PagedVector/PagedVector.hpp>
#include <Join/NestedLoopJoin/NLJSlice.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/DefaultTimeBasedSliceStore.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/ConstantPressureSensor.hpp>
#include <SliceStore/Spill/InMemoryStorageBackend.hpp>
#include <SliceStore/Spill/NLJSliceStateSerializer.hpp>
#include <SliceStore/Spill/PressureSpillPolicy.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/SliceStateSerializerRegistry.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/SpillingTimeBasedSliceStore.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <SliceCacheConfiguration.hpp>

namespace NES
{

namespace
{
constexpr uint64_t WindowSize = 100;
constexpr uint64_t WindowSlide = 100;
constexpr uint64_t NumWorkerThreads = 2;
}

class SpillingTimeBasedSliceStoreTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite()
    {
        Logger::setupLogging("SpillingTimeBasedSliceStoreTest.log", LogLevel::LOG_DEBUG);
        /// Ensure the NLJSliceStateSerializer's static-init registrar is pulled in by the linker.
        (void)forceLinkNLJSerializer();
    }

    void SetUp() override
    {
        Testing::BaseUnitTest::SetUp();
        bufferManager = BufferManager::create(/*bufferSize*/ 4096, /*numOfBuffers*/ 64);
    }

    /// Builds a Spilling store decorating a DefaultTimeBasedSliceStore. The reactive policy fires
    /// at high pressure; the ConstantPressureSensor lets the test drive the decision.
    std::unique_ptr<SpillingTimeBasedSliceStore> makeStore(std::shared_ptr<InMemoryStorageBackend> backend, double constantPressure)
    {
        auto inner = std::make_unique<DefaultTimeBasedSliceStore>(WindowSize, WindowSlide, SliceCacheConfiguration{});
        auto policy = std::make_unique<PressureSpillPolicy>(/*high*/ 0.8);
        auto sensor = std::make_unique<ConstantPressureSensor>(constantPressure);
        auto* serializer = SliceStateSerializerRegistry::instance().lookup("NLJSlice");
        auto store = std::make_unique<SpillingTimeBasedSliceStore>(
            std::move(inner), std::move(policy), std::move(backend), std::move(sensor), *bufferManager, *serializer);
        /// DefaultTimeBasedSliceStore::getAllNonTriggeredSlices precondition: at least one input pipeline registered.
        store->incrementNumberOfInputPipelines();
        return store;
    }

    /// Allocates an unpooled buffer, fills it with a recognisable byte pattern, sets the tuple count.
    TupleBuffer makeFilledBuffer(uint64_t tupleCount, uint8_t fillByte)
    {
        auto opt = bufferManager->getUnpooledBuffer(64);
        EXPECT_TRUE(opt.has_value());
        auto buffer = std::move(opt.value());
        auto span = buffer.getAvailableMemoryArea<uint8_t>();
        for (auto& b : span)
        {
            b = fillByte;
        }
        buffer.setNumberOfTuples(tupleCount);
        return buffer;
    }

    std::shared_ptr<AbstractBufferProvider> bufferManager;
};

namespace
{
/// Records what the store passes to observe(), so the (now, globalWatermark) contract is testable.
class RecordingSpillPolicy final : public SpillPolicy
{
public:
    [[nodiscard]] SpillDecision decide(const SliceSpillContext& ctx, double /*memoryPressure*/) const override
    {
        lastContextNow = ctx.now;
        return SpillDecision::Keep;
    }

    void observe(Timestamp now, Timestamp globalWatermark) noexcept override
    {
        lastNow = now;
        lastWatermark = globalWatermark;
        ++observeCalls;
    }

    Timestamp lastNow{Timestamp::INVALID_VALUE};
    Timestamp lastWatermark{Timestamp::INVALID_VALUE};
    mutable Timestamp lastContextNow{Timestamp::INVALID_VALUE};
    uint64_t observeCalls{0};
};
}

/// Regression: the store used to call observe(watermark, wallClock) against a (now, watermark)
/// signature, training the predictor on transposed axes. It must also hand decide() a wall-clock
/// `now`, because a predictive policy compares it against a wall-clock trigger estimate.
TEST_F(SpillingTimeBasedSliceStoreTest, ObservePassesWallClockAsNowAndWatermarkAsWatermark)
{
    constexpr uint64_t FakeWallClock = 9999;
    auto policy = std::make_unique<RecordingSpillPolicy>();
    auto* policyPtr = policy.get();

    auto inner = std::make_unique<DefaultTimeBasedSliceStore>(WindowSize, WindowSlide, SliceCacheConfiguration{});
    auto* serializer = SliceStateSerializerRegistry::instance().lookup("NLJSlice");
    auto store = std::make_unique<SpillingTimeBasedSliceStore>(
        std::move(inner),
        std::move(policy),
        std::make_shared<InMemoryStorageBackend>(),
        std::make_unique<ConstantPressureSensor>(0.95),
        *bufferManager,
        *serializer);
    store->incrementNumberOfInputPipelines();
    store->setWallClockSourceForTesting([] { return Timestamp{FakeWallClock}; });

    store->getSlicesOrCreate(
        Timestamp{50},
        [](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(start, end, NumWorkerThreads));
            return out;
        });

    store->garbageCollectSlicesAndWindows(Timestamp{100});

    ASSERT_EQ(policyPtr->observeCalls, 1U);
    EXPECT_EQ(policyPtr->lastNow, Timestamp{FakeWallClock});
    EXPECT_EQ(policyPtr->lastWatermark, Timestamp{100});
    EXPECT_EQ(policyPtr->lastContextNow, Timestamp{FakeWallClock});
}

TEST_F(SpillingTimeBasedSliceStoreTest, LowPressureKeepsAllSlicesResident)
{
    auto backend = std::make_shared<InMemoryStorageBackend>();
    auto store = makeStore(backend, /*pressure*/ 0.1);

    /// Insert a slice via getSlicesOrCreate; populate its left PagedVector with one page.
    auto created = store->getSlicesOrCreate(
        Timestamp{50},
        [](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(start, end, NumWorkerThreads));
            return out;
        });
    ASSERT_EQ(created.size(), 1U);
    auto& slice = static_cast<NLJSlice&>(*created.front());
    slice.getPagedVectorRefLeft(WorkerThreadId{0})->adoptPages({makeFilledBuffer(3, 0xAA)});

    /// Drive a GC tick at an early watermark (below window end) — under low pressure nothing spills.
    store->garbageCollectSlicesAndWindows(Timestamp{50});
    EXPECT_EQ(store->numSpilledSlices(), 0U);
    EXPECT_FALSE(store->isSliceSpilled(SliceEnd{100}));
}

TEST_F(SpillingTimeBasedSliceStoreTest, HighPressureSpillsSliceAndRestoreOnProbeRoundTrips)
{
    auto backend = std::make_shared<InMemoryStorageBackend>();
    auto store = makeStore(backend, /*pressure*/ 0.95);

    auto created = store->getSlicesOrCreate(
        Timestamp{50},
        [](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(start, end, NumWorkerThreads));
            return out;
        });
    ASSERT_EQ(created.size(), 1U);
    auto& slice = static_cast<NLJSlice&>(*created.front());

    /// Populate two pages on the left side under worker 0 with distinguishable fill bytes.
    slice.getPagedVectorRefLeft(WorkerThreadId{0})->adoptPages({makeFilledBuffer(3, 0xAB), makeFilledBuffer(5, 0xCD)});
    const auto preSpillTuples = slice.getNumberOfTuplesLeft();
    ASSERT_EQ(preSpillTuples, 8U);

    /// GC tick — under high pressure the slice is spilled.
    store->garbageCollectSlicesAndWindows(Timestamp{50});
    EXPECT_EQ(store->numSpilledSlices(), 1U);
    EXPECT_TRUE(store->isSliceSpilled(SliceEnd{100}));
    EXPECT_EQ(slice.getNumberOfTuplesLeft(), 0U);
    EXPECT_GE(backend->numStoredObjects(), 2U);

    /// Probe-side access triggers a synchronous restore.
    auto restored = store->getSliceBySliceEnd(SliceEnd{100});
    ASSERT_TRUE(restored.has_value());
    EXPECT_FALSE(store->isSliceSpilled(SliceEnd{100}));
    EXPECT_EQ(static_cast<NLJSlice&>(*restored.value()).getNumberOfTuplesLeft(), preSpillTuples);
}

/// Regression: the decorator used to build its GC candidate set from slices it saw pass through its
/// own getSlicesOrCreate. The JIT-compiled build path does not go through the decorator — it holds a
/// SliceStoreRef bound to the concrete inner store — so in a real query the candidate set was always
/// empty and nothing was ever spilled at any pressure. Creating the slice directly on the inner store
/// reproduces that bypass; the decorator must still find and spill it.
TEST_F(SpillingTimeBasedSliceStoreTest, SpillsSlicesCreatedDirectlyOnTheInnerStore)
{
    auto backend = std::make_shared<InMemoryStorageBackend>();
    auto inner = std::make_unique<DefaultTimeBasedSliceStore>(WindowSize, WindowSlide, SliceCacheConfiguration{});
    auto* innerRaw = inner.get();
    auto* serializer = SliceStateSerializerRegistry::instance().lookup("NLJSlice");
    auto store = std::make_unique<SpillingTimeBasedSliceStore>(
        std::move(inner),
        std::make_unique<PressureSpillPolicy>(/*high*/ 0.8),
        backend,
        std::make_unique<ConstantPressureSensor>(0.95),
        *bufferManager,
        *serializer);
    store->incrementNumberOfInputPipelines();

    /// Bypass the decorator exactly as the SliceStoreRef does: straight to the inner store.
    auto created = innerRaw->getSlicesOrCreate(
        Timestamp{50},
        [](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(start, end, NumWorkerThreads));
            return out;
        });
    ASSERT_EQ(created.size(), 1U);
    auto& slice = static_cast<NLJSlice&>(*created.front());
    slice.getPagedVectorRefLeft(WorkerThreadId{0})->adoptPages({makeFilledBuffer(4, 0x5A)});
    ASSERT_EQ(slice.getNumberOfTuplesLeft(), 4U);

    store->garbageCollectSlicesAndWindows(Timestamp{100});

    EXPECT_EQ(store->numSpilledSlices(), 1U);
    EXPECT_TRUE(store->isSliceSpilled(SliceEnd{100}));
    EXPECT_EQ(slice.getNumberOfTuplesLeft(), 0U);
    EXPECT_EQ(store->statistics().spills.load(), 1U);
}

/// A serializer that refuses a slice (HJSliceStateSerializer rejects hash maps with var-sized pages)
/// must leave the slice resident and the query running, not propagate out of the GC tick.
TEST_F(SpillingTimeBasedSliceStoreTest, UnspillableSliceStaysResidentAndDoesNotThrow)
{
    class RefusingSerializer final : public SliceStateSerializer
    {
    public:
        std::future<std::expected<SpilledSliceHandle, IoError>> spill(Slice&, StorageBackend&, AbstractBufferProvider&) override
        {
            std::promise<std::expected<SpilledSliceHandle, IoError>> p;
            p.set_value(std::unexpected{IoError{IoErrorCode::TransientIo, "refused"}});
            return p.get_future();
        }

        std::future<std::expected<void, IoError>>
        restore(Slice&, const SpilledSliceHandle&, StorageBackend&, AbstractBufferProvider&) override
        {
            std::promise<std::expected<void, IoError>> p;
            p.set_value({});
            return p.get_future();
        }

        [[nodiscard]] uint64_t residentBytes(const Slice&) const noexcept override { return 1024; }
    };

    RefusingSerializer refusing;
    auto inner = std::make_unique<DefaultTimeBasedSliceStore>(WindowSize, WindowSlide, SliceCacheConfiguration{});
    auto store = std::make_unique<SpillingTimeBasedSliceStore>(
        std::move(inner),
        std::make_unique<PressureSpillPolicy>(/*high*/ 0.8),
        std::make_shared<InMemoryStorageBackend>(),
        std::make_unique<ConstantPressureSensor>(0.95),
        *bufferManager,
        refusing);
    store->incrementNumberOfInputPipelines();

    auto created = store->getSlicesOrCreate(
        Timestamp{50},
        [](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(start, end, NumWorkerThreads));
            return out;
        });
    auto& slice = static_cast<NLJSlice&>(*created.front());
    slice.getPagedVectorRefLeft(WorkerThreadId{0})->adoptPages({makeFilledBuffer(7, 0x11)});

    EXPECT_NO_THROW(store->garbageCollectSlicesAndWindows(Timestamp{100}));
    EXPECT_EQ(store->numSpilledSlices(), 0U);
    EXPECT_EQ(slice.getNumberOfTuplesLeft(), 7U) << "a refused spill must leave the slice intact";
    EXPECT_EQ(store->statistics().spillFailures.load(), 1U);
}

TEST_F(SpillingTimeBasedSliceStoreTest, SpilledSlicesAreNotDoubleSpilled)
{
    auto backend = std::make_shared<InMemoryStorageBackend>();
    auto store = makeStore(backend, /*pressure*/ 0.95);

    auto created = store->getSlicesOrCreate(
        Timestamp{50},
        [](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(start, end, NumWorkerThreads));
            return out;
        });
    auto& slice = static_cast<NLJSlice&>(*created.front());
    slice.getPagedVectorRefLeft(WorkerThreadId{0})->adoptPages({makeFilledBuffer(1, 0x01)});

    store->garbageCollectSlicesAndWindows(Timestamp{50});
    const auto firstSpillObjects = backend->numStoredObjects();
    ASSERT_EQ(store->numSpilledSlices(), 1U);

    /// Second GC tick: the slice is already in handles, so spill should be skipped — no new objects.
    store->garbageCollectSlicesAndWindows(Timestamp{60});
    EXPECT_EQ(backend->numStoredObjects(), firstSpillObjects);
    EXPECT_EQ(store->numSpilledSlices(), 1U);
}

}
