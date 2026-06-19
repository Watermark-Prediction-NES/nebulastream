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
#include <SliceStore/Spill/SliceStateSerializerRegistry.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
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
