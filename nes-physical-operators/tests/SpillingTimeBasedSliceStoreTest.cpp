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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <typeindex>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Interface/PagedVector/PagedVector.hpp>
#include <Join/NestedLoopJoin/NLJSlice.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/Allocator/NesDefaultMemoryAllocator.hpp>
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
constexpr uint32_t POOLED_BUFFER_SIZE = 4096;
constexpr size_t TOTAL_MEMORY_IN_BYTES = 640 * POOLED_BUFFER_SIZE;
constexpr double UNPOOLED_MEMORY_FRACTION = 0.9;
/// NLJSlice needs a tuple size to lay out its PagedVector pages; the tests only count tuples, so any
/// size that leaves room for a page header works.
constexpr uint64_t TupleSize = 8;
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
        bufferManager = BufferManager::create(
            TOTAL_MEMORY_IN_BYTES,
            UNPOOLED_MEMORY_FRACTION,
            BufferAlignment{64},
            POOLED_BUFFER_SIZE,
            std::make_shared<NesDefaultMemoryAllocator>());
    }

    /// Builds a Spilling store decorating a DefaultTimeBasedSliceStore. The reactive policy fires
    /// at high pressure; the ConstantPressureSensor lets the test drive the decision.
    std::unique_ptr<SpillingTimeBasedSliceStore> makeStore(std::shared_ptr<InMemoryStorageBackend> backend, double constantPressure)
    {
        return makeStore(std::make_unique<PressureSpillPolicy>(/*high*/ 0.8), std::move(backend), nullptr, nullptr, constantPressure);
    }

    /// Full tier wiring. `disk` is where PressureSpillPolicy sends everything; the two compressed slots
    /// stay null unless a test needs them, which mirrors how the factory wires a non-tiered policy.
    std::unique_ptr<SpillingTimeBasedSliceStore> makeStore(
        std::unique_ptr<SpillPolicy> policy,
        std::shared_ptr<StorageBackend> disk,
        std::shared_ptr<StorageBackend> compressedRam,
        std::shared_ptr<StorageBackend> compressedDisk,
        double constantPressure)
    {
        auto inner = std::make_unique<DefaultTimeBasedSliceStore>(WindowSize, WindowSlide, SliceCacheConfiguration{});
        auto sensor = std::make_unique<ConstantPressureSensor>(constantPressure);
        auto* serializer = SliceStateSerializerRegistry::instance().lookup("NLJSlice");
        SpillingTimeBasedSliceStore::TierBackends backends{};
        backends[static_cast<std::size_t>(SliceTier::Disk)] = std::move(disk);
        backends[static_cast<std::size_t>(SliceTier::CompressedRam)] = std::move(compressedRam);
        backends[static_cast<std::size_t>(SliceTier::CompressedDisk)] = std::move(compressedDisk);
        auto store = std::make_unique<SpillingTimeBasedSliceStore>(
            std::move(inner), std::move(policy), std::move(backends), std::move(sensor), *bufferManager, *serializer);
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
        [this](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(*bufferManager, start, end, NumWorkerThreads, TupleSize, TupleSize));
            return out;
        });
    ASSERT_EQ(created.size(), 1U);
    auto& slice = static_cast<NLJSlice&>(*created.front());
    PagedVector::load(*slice.getPagedVectorRefLeft(WorkerThreadId{0})).adoptPages({makeFilledBuffer(3, 0xAA)});

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
        [this](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(*bufferManager, start, end, NumWorkerThreads, TupleSize, TupleSize));
            return out;
        });
    ASSERT_EQ(created.size(), 1U);
    auto& slice = static_cast<NLJSlice&>(*created.front());

    /// Populate two pages on the left side under worker 0 with distinguishable fill bytes.
    PagedVector::load(*slice.getPagedVectorRefLeft(WorkerThreadId{0})).adoptPages({makeFilledBuffer(3, 0xAB), makeFilledBuffer(5, 0xCD)});
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
        [this](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(*bufferManager, start, end, NumWorkerThreads, TupleSize, TupleSize));
            return out;
        });
    auto& slice = static_cast<NLJSlice&>(*created.front());
    PagedVector::load(*slice.getPagedVectorRefLeft(WorkerThreadId{0})).adoptPages({makeFilledBuffer(1, 0x01)});

    store->garbageCollectSlicesAndWindows(Timestamp{50});
    const auto firstSpillObjects = backend->numStoredObjects();
    ASSERT_EQ(store->numSpilledSlices(), 1U);

    /// Second GC tick: the slice is re-evaluated now (that is what makes demotion possible), but the
    /// reactive policy names the tier it is already in, so nothing moves and no new objects appear.
    store->garbageCollectSlicesAndWindows(Timestamp{60});
    EXPECT_EQ(backend->numStoredObjects(), firstSpillObjects);
    EXPECT_EQ(store->numSpilledSlices(), 1U);
}

namespace
{
/// Names a tier from a script, one entry per decide() call, so a test can drive an exact sequence of
/// tier moves without depending on a predictor's arithmetic.
class ScriptedTierPolicy final : public SpillPolicy
{
public:
    explicit ScriptedTierPolicy(std::vector<SliceTier> script) : script(std::move(script)) { }

    [[nodiscard]] SliceTier decide(const SliceSpillContext& ctx, double /*memoryPressure*/) const override
    {
        return next < script.size() ? script[next++] : ctx.currentTier;
    }

    void observe(Timestamp now, Timestamp globalWatermark) noexcept override
    {
        observedNow = now;
        observedWatermark = globalWatermark;
    }

    Timestamp observedNow{Timestamp::INVALID_VALUE};
    Timestamp observedWatermark{Timestamp::INVALID_VALUE};

private:
    std::vector<SliceTier> script;
    mutable std::size_t next{0};
};
}

TEST_F(SpillingTimeBasedSliceStoreTest, ObserveReceivesWallClockNotWatermark)
{
    auto policy = std::make_unique<ScriptedTierPolicy>(std::vector<SliceTier>{});
    auto* policyPtr = policy.get();
    auto store = makeStore(std::move(policy), std::make_shared<InMemoryStorageBackend>(), nullptr, nullptr, /*pressure*/ 0.1);
    store->setWallClockSourceForTesting([] { return Timestamp{999999}; });

    store->garbageCollectSlicesAndWindows(Timestamp{50});

    /// The predictor learns (watermark, wallClock) pairs; passing the watermark as both would make it
    /// fit a slope against itself and render every horizon meaningless.
    EXPECT_EQ(policyPtr->observedNow, Timestamp{999999});
    EXPECT_EQ(policyPtr->observedWatermark, Timestamp{50});
}

TEST_F(SpillingTimeBasedSliceStoreTest, TierMoveRoundTripsSliceThroughEveryTier)
{
    auto ram = std::make_shared<InMemoryStorageBackend>();
    auto zdisk = std::make_shared<InMemoryStorageBackend>();
    auto disk = std::make_shared<InMemoryStorageBackend>();
    /// Resident -> CompressedRam -> CompressedDisk -> Disk -> Resident, one hop per GC tick.
    auto policy = std::make_unique<ScriptedTierPolicy>(
        std::vector<SliceTier>{SliceTier::CompressedRam, SliceTier::CompressedDisk, SliceTier::Disk, SliceTier::Resident});
    auto store = makeStore(std::move(policy), disk, ram, zdisk, /*pressure*/ 0.95);

    auto created = store->getSlicesOrCreate(
        Timestamp{50},
        [this](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(*bufferManager, start, end, NumWorkerThreads, TupleSize, TupleSize));
            return out;
        });
    ASSERT_EQ(created.size(), 1U);
    auto& slice = static_cast<NLJSlice&>(*created.front());
    PagedVector::load(*slice.getPagedVectorRefLeft(WorkerThreadId{0})).adoptPages({makeFilledBuffer(3, 0xAB), makeFilledBuffer(5, 0xCD)});
    const auto preSpillTuples = slice.getNumberOfTuplesLeft();
    ASSERT_EQ(preSpillTuples, 8U);

    store->garbageCollectSlicesAndWindows(Timestamp{50});
    EXPECT_EQ(store->tierOf(SliceEnd{100}), SliceTier::CompressedRam);
    EXPECT_GE(ram->numStoredObjects(), 2U);

    /// Lateral move: the bytes leave the RAM tier and land on the compressed-disk tier.
    store->garbageCollectSlicesAndWindows(Timestamp{51});
    EXPECT_EQ(store->tierOf(SliceEnd{100}), SliceTier::CompressedDisk);
    EXPECT_GE(zdisk->numStoredObjects(), 2U);

    store->garbageCollectSlicesAndWindows(Timestamp{52});
    EXPECT_EQ(store->tierOf(SliceEnd{100}), SliceTier::Disk);
    EXPECT_GE(disk->numStoredObjects(), 2U);

    /// Eager promotion: the slice comes back without the probe asking for it, and its bytes survived
    /// every hop — which is only true if each restore read through the tier that actually held them.
    store->garbageCollectSlicesAndWindows(Timestamp{53});
    EXPECT_EQ(store->tierOf(SliceEnd{100}), SliceTier::Resident);
    EXPECT_FALSE(store->isSliceSpilled(SliceEnd{100}));
    EXPECT_EQ(slice.getNumberOfTuplesLeft(), preSpillTuples);
}

TEST_F(SpillingTimeBasedSliceStoreTest, RestoreReadsBackThroughTheTierThatHoldsTheBytes)
{
    auto ram = std::make_shared<InMemoryStorageBackend>();
    auto disk = std::make_shared<InMemoryStorageBackend>();
    auto policy = std::make_unique<ScriptedTierPolicy>(std::vector<SliceTier>{SliceTier::CompressedRam});
    auto store = makeStore(std::move(policy), disk, ram, nullptr, /*pressure*/ 0.95);

    auto created = store->getSlicesOrCreate(
        Timestamp{50},
        [this](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(*bufferManager, start, end, NumWorkerThreads, TupleSize, TupleSize));
            return out;
        });
    auto& slice = static_cast<NLJSlice&>(*created.front());
    PagedVector::load(*slice.getPagedVectorRefLeft(WorkerThreadId{0})).adoptPages({makeFilledBuffer(7, 0xEE)});

    store->garbageCollectSlicesAndWindows(Timestamp{50});
    ASSERT_EQ(store->tierOf(SliceEnd{100}), SliceTier::CompressedRam);
    /// The Disk backend must be untouched — reading the slice back through it would find nothing.
    EXPECT_EQ(disk->numStoredObjects(), 0U);

    auto restored = store->getSliceBySliceEnd(SliceEnd{100});
    ASSERT_TRUE(restored.has_value());
    EXPECT_EQ(static_cast<NLJSlice&>(*restored.value()).getNumberOfTuplesLeft(), 7U);
}

TEST_F(SpillingTimeBasedSliceStoreTest, UnwiredTierLeavesSliceInPlace)
{
    auto disk = std::make_shared<InMemoryStorageBackend>();
    /// CompressedRam is not wired -- what the factory does for a non-tiered policy, and also for a
    /// tiered one with compression disabled, where the ladder degrades to Resident <-> Disk. The policy
    /// may still name the tier; naming an unwired tier must be inert, not a crash or a lost slice.
    auto policy = std::make_unique<ScriptedTierPolicy>(std::vector<SliceTier>{SliceTier::CompressedRam});
    auto store = makeStore(std::move(policy), disk, nullptr, nullptr, /*pressure*/ 0.95);

    auto created = store->getSlicesOrCreate(
        Timestamp{50},
        [this](SliceStart start, SliceEnd end) -> std::vector<std::shared_ptr<Slice>>
        {
            std::vector<std::shared_ptr<Slice>> out;
            out.push_back(std::make_shared<NLJSlice>(*bufferManager, start, end, NumWorkerThreads, TupleSize, TupleSize));
            return out;
        });
    auto& slice = static_cast<NLJSlice&>(*created.front());
    PagedVector::load(*slice.getPagedVectorRefLeft(WorkerThreadId{0})).adoptPages({makeFilledBuffer(2, 0x11)});

    store->garbageCollectSlicesAndWindows(Timestamp{50});
    EXPECT_EQ(store->numSpilledSlices(), 0U);
    EXPECT_EQ(slice.getNumberOfTuplesLeft(), 2U);
}

}
