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

/// Round-trips real slices through ReducibleSlice. BufferTreeCodecTest already establishes that an
/// arbitrary buffer tree survives encode/decode; what is left to prove here is that the slices hand the
/// codec the right buffers and put them back in the right slots — including the slots no build worker
/// ever touched, which are the ones a lazily-allocating slice is most likely to get wrong.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <Identifiers/Identifiers.hpp>
#include <Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Interface/PagedVector/PagedVector.hpp>
#include <Join/HashJoin/HJSlice.hpp>
#include <Join/NestedLoopJoin/NLJSlice.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/Allocator/NesDefaultMemoryAllocator.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <HashMapSlice.hpp>

namespace NES
{
namespace
{
constexpr uint32_t POOLED_BUFFER_SIZE = 4096;
constexpr uint32_t NUMBER_OF_POOLED_BUFFERS = 1024;
constexpr BufferAlignment BUFFER_ALIGNMENT{64};
constexpr double UNPOOLED_MEMORY_FRACTION = 0.9;
constexpr size_t TOTAL_MEMORY_IN_BYTES = 20 * static_cast<size_t>(NUMBER_OF_POOLED_BUFFERS) * POOLED_BUFFER_SIZE;

constexpr uint64_t NUMBER_OF_WORKER_THREADS = 4;
constexpr uint64_t TUPLE_SIZE = 16;
constexpr SliceStart SLICE_START{0};
constexpr SliceEnd SLICE_END{1000};

/// ChainedHashMap parameters. The page size is small on purpose so that the inserts below spill across
/// several pages, which is what exercises the page walk in rebuildChains().
constexpr uint64_t KEY_SIZE = 8;
constexpr uint64_t VALUE_SIZE = 8;
constexpr uint64_t NUMBER_OF_BUCKETS = 16;
constexpr uint64_t HASH_MAP_PAGE_SIZE = 256;

std::shared_ptr<AbstractBufferProvider> makeBufferManager()
{
    return BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NesDefaultMemoryAllocator>());
}

/// One record's worth of bytes, derived from a seed so that every record in a test is distinguishable.
std::vector<std::byte> recordBytes(const uint64_t seed)
{
    std::vector<std::byte> record(TUPLE_SIZE);
    for (uint64_t i = 0; i < TUPLE_SIZE; ++i)
    {
        record[i] = static_cast<std::byte>((seed * 31 + i) % 251);
    }
    return record;
}

/// Appends one record to a PagedVector by hand. PagedVectorRef is the supported writer, but it only
/// exists in traced code; here the fixed-size layout is written directly, which is all these tests need.
void appendRecord(const TupleBuffer& mainBuffer, AbstractBufferProvider& bufferProvider, const uint64_t seed)
{
    auto pagedVector = PagedVector::load(mainBuffer);
    pagedVector.appendPageIfFull(&bufferProvider);
    const ChildBufferIndex lastPageIndex{static_cast<uint32_t>(pagedVector.getNumberOfPages() - 1)};
    auto pageBuffer = mainBuffer.loadChildBuffer(lastPageIndex);

    const auto numberOfTuples = pageBuffer.getNumberOfTuples();
    const auto offset = PagedVector::Page::getHeaderSize() + (numberOfTuples * TUPLE_SIZE);
    const auto record = recordBytes(seed);
    std::memcpy(pageBuffer.getAvailableMemoryArea().data() + offset, record.data(), record.size());
    pageBuffer.setNumberOfTuples(numberOfTuples + 1);
}

/// Everything a PagedVector holds, flattened: page contents plus the contents of every child buffer
/// hanging off a page. The latter is where variable-sized payloads live, so comparing this is what
/// catches a codec that drops them.
struct PagedVectorSnapshot
{
    uint64_t numberOfPages{0};
    std::vector<std::vector<std::byte>> pages;
    std::vector<std::vector<std::vector<std::byte>>> pageChildren;

    bool operator==(const PagedVectorSnapshot&) const = default;
};

std::vector<std::byte> bytesOf(const TupleBuffer& buffer)
{
    const auto memory = buffer.getAvailableMemoryArea();
    return {memory.begin(), memory.end()};
}

PagedVectorSnapshot snapshotPagedVector(const TupleBuffer& mainBuffer)
{
    PagedVectorSnapshot snapshot;
    const auto pagedVector = PagedVector::load(mainBuffer);
    snapshot.numberOfPages = pagedVector.getNumberOfPages();
    for (uint64_t pageIdx = 0; pageIdx < snapshot.numberOfPages; ++pageIdx)
    {
        const auto page = mainBuffer.loadChildBuffer(ChildBufferIndex{static_cast<uint32_t>(pageIdx)});
        snapshot.pages.push_back(bytesOf(page));

        std::vector<std::vector<std::byte>> children;
        for (uint32_t childIdx = 0; childIdx < page.getNumberOfChildBuffers(); ++childIdx)
        {
            children.push_back(bytesOf(page.loadChildBuffer(ChildBufferIndex{childIdx})));
        }
        snapshot.pageChildren.push_back(std::move(children));
    }
    return snapshot;
}

std::vector<PagedVectorSnapshot> snapshotSide(const NLJSlice& slice, const JoinBuildSideType side)
{
    std::vector<PagedVectorSnapshot> snapshots;
    for (uint64_t i = 0; i < NUMBER_OF_WORKER_THREADS; ++i)
    {
        const auto* const mainBuffer = slice.getPagedVectorTupleBufferRef(WorkerThreadId(i), side);
        snapshots.push_back(snapshotPagedVector(*mainBuffer));
    }
    return snapshots;
}

/// Contents of one chain, head-first, as (hash, first 8 payload bytes) pairs. Order matters: chain order
/// is probe order, and probe order is the row order of a join's output.
using ChainSnapshot = std::vector<std::pair<uint64_t, uint64_t>>;

uint64_t payloadOf(ChainedHashMapEntry* entry)
{
    uint64_t payload = 0;
    /// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-type-reinterpret-cast)
    std::memcpy(&payload, reinterpret_cast<std::byte*>(entry) + sizeof(ChainedHashMapEntry), sizeof(payload));
    return payload;
}

void writePayload(ChainedHashMapEntry* entry, const uint64_t payload)
{
    /// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-type-reinterpret-cast)
    std::memcpy(reinterpret_cast<std::byte*>(entry) + sizeof(ChainedHashMapEntry), &payload, sizeof(payload));
}

std::vector<ChainSnapshot> snapshotChains(ChainedHashMap& hashMap)
{
    std::vector<ChainSnapshot> snapshot;
    snapshot.reserve(hashMap.getNumberOfChains());
    for (uint64_t pos = 0; pos < hashMap.getNumberOfChains(); ++pos)
    {
        ChainSnapshot chain;
        for (auto* entry = hashMap.getChain(pos); entry != nullptr; entry = entry->next)
        {
            chain.emplace_back(entry->hash, payloadOf(entry));
        }
        snapshot.push_back(std::move(chain));
    }
    return snapshot;
}

CreateNewHJSliceArgs hjSliceArgs(AbstractBufferProvider& bufferProvider, const JoinBuildSideType side)
{
    return CreateNewHJSliceArgs{KEY_SIZE, VALUE_SIZE, HASH_MAP_PAGE_SIZE, NUMBER_OF_BUCKETS, &bufferProvider, side};
}
}

class ReducibleSliceTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("ReducibleSliceTest.log", LogLevel::LOG_DEBUG); }

    void SetUp() override
    {
        Testing::BaseUnitTest::SetUp();
        bufferManager = makeBufferManager();
    }

    std::shared_ptr<AbstractBufferProvider> bufferManager;

    /// Fills both sides across all worker threads, so the test covers more than the slot at index 0.
    [[nodiscard]] std::unique_ptr<NLJSlice> makePopulatedNLJSlice(const uint64_t recordsPerSlot) const
    {
        auto slice = std::make_unique<NLJSlice>(*bufferManager, SLICE_START, SLICE_END, NUMBER_OF_WORKER_THREADS, TUPLE_SIZE, TUPLE_SIZE);
        for (uint64_t worker = 0; worker < NUMBER_OF_WORKER_THREADS; ++worker)
        {
            for (const auto side : {JoinBuildSideType::Left, JoinBuildSideType::Right})
            {
                const auto* const mainBuffer = slice->getPagedVectorTupleBufferRef(WorkerThreadId(worker), side);
                for (uint64_t record = 0; record < recordsPerSlot; ++record)
                {
                    appendRecord(*mainBuffer, *bufferManager, (worker * 1000) + record + (side == JoinBuildSideType::Right ? 500 : 0));
                }
            }
        }
        return slice;
    }
};

/// ---------------------------------------------------------------------------------------------
/// NLJSlice
/// ---------------------------------------------------------------------------------------------

TEST_F(ReducibleSliceTest, NLJSliceRoundTripsEveryRecordOnBothSides)
{
    /// Enough records to need several pages per slot at a 4 KiB page size.
    auto slice = makePopulatedNLJSlice(600);
    const auto leftBefore = snapshotSide(*slice, JoinBuildSideType::Left);
    const auto rightBefore = snapshotSide(*slice, JoinBuildSideType::Right);
    const auto bytesBefore = slice->residentBytes();
    ASSERT_GT(bytesBefore, 0U);

    std::vector<std::byte> encoded;
    slice->serializeState(encoded);

    EXPECT_TRUE(slice->isReduced());
    EXPECT_EQ(slice->residentBytes(), 0U) << "reduction must actually release the buffers, not just copy them";

    slice->deserializeState(encoded, *bufferManager);

    EXPECT_FALSE(slice->isReduced());
    EXPECT_EQ(slice->residentBytes(), bytesBefore);
    EXPECT_EQ(snapshotSide(*slice, JoinBuildSideType::Left), leftBefore);
    EXPECT_EQ(snapshotSide(*slice, JoinBuildSideType::Right), rightBefore);
}

TEST_F(ReducibleSliceTest, NLJSliceRoundTripsVariableSizedPayloads)
{
    auto slice = makePopulatedNLJSlice(10);

    /// Variable-sized data lives in buffers attached to the page holding the record that references it,
    /// and records address them by ChildBufferIndex. Attaching one directly is enough to prove the codec
    /// preserves both the payload and the index it is reached by.
    const auto* const mainBuffer = slice->getPagedVectorTupleBufferRef(WorkerThreadId(0), JoinBuildSideType::Left);
    auto page = mainBuffer->loadChildBuffer(ChildBufferIndex{0});
    for (uint64_t i = 0; i < 3; ++i)
    {
        auto varSized = bufferManager->getUnpooledBuffer(64);
        ASSERT_TRUE(varSized.has_value());
        const auto payload = recordBytes(9000 + i);
        std::memcpy(varSized->getAvailableMemoryArea().data(), payload.data(), payload.size());
        const auto index = page.storeChildBuffer(varSized.value());
        ASSERT_EQ(index.getRawValue(), i) << "child indices must be handed out in insertion order";
    }

    const auto before = snapshotSide(*slice, JoinBuildSideType::Left);
    ASSERT_EQ(before[0].pageChildren[0].size(), 3U);

    std::vector<std::byte> encoded;
    slice->serializeState(encoded);
    slice->deserializeState(encoded, *bufferManager);

    EXPECT_EQ(snapshotSide(*slice, JoinBuildSideType::Left), before);
}

TEST_F(ReducibleSliceTest, NLJSliceRoundTripsSlotsThatWereNeverWrittenTo)
{
    /// A slice whose worker threads never all ran. The empty PagedVectors are still initialised main
    /// buffers, so they have to come back as valid empty vectors rather than as absent slots.
    auto slice = makePopulatedNLJSlice(0);
    const auto before = snapshotSide(*slice, JoinBuildSideType::Left);

    std::vector<std::byte> encoded;
    slice->serializeState(encoded);
    slice->deserializeState(encoded, *bufferManager);

    EXPECT_EQ(snapshotSide(*slice, JoinBuildSideType::Left), before);
    for (uint64_t worker = 0; worker < NUMBER_OF_WORKER_THREADS; ++worker)
    {
        const auto* const mainBuffer = slice->getPagedVectorTupleBufferRef(WorkerThreadId(worker), JoinBuildSideType::Left);
        const auto pagedVector = PagedVector::load(*mainBuffer);
        EXPECT_EQ(pagedVector.getStatus(), PagedVector::VALID_PV);
        EXPECT_EQ(pagedVector.getTotalNumberOfRecords(), 0U);
    }
}

TEST_F(ReducibleSliceTest, NLJSliceRoundTripsAfterItsPagedVectorsWereCombined)
{
    /// combinePagedVectors() moves every page onto the slot at index 0 and erases the rest, so the slot
    /// vector this reduction sees is a different shape from the one the slice was built with.
    auto slice = makePopulatedNLJSlice(100);
    slice->combinePagedVectors();

    const auto tuplesLeft = slice->getNumberOfTuplesLeft();
    const auto tuplesRight = slice->getNumberOfTuplesRight();
    const auto* const combinedLeft = slice->getPagedVectorTupleBufferRef(WorkerThreadId(0), JoinBuildSideType::Left);
    const auto before = snapshotPagedVector(*combinedLeft);

    std::vector<std::byte> encoded;
    slice->serializeState(encoded);
    slice->deserializeState(encoded, *bufferManager);

    EXPECT_EQ(slice->getNumberOfTuplesLeft(), tuplesLeft);
    EXPECT_EQ(slice->getNumberOfTuplesRight(), tuplesRight);
    const auto* const afterBuffer = slice->getPagedVectorTupleBufferRef(WorkerThreadId(0), JoinBuildSideType::Left);
    EXPECT_EQ(snapshotPagedVector(*afterBuffer), before);
}

TEST_F(ReducibleSliceTest, SerializingAnAlreadyReducedNLJSliceDoesNothing)
{
    auto slice = makePopulatedNLJSlice(10);

    std::vector<std::byte> encoded;
    slice->serializeState(encoded);
    const auto sizeAfterFirst = encoded.size();

    /// A second reduction must not append a second copy: the manager would then decode only the first
    /// one and leave the rest of the stream dangling.
    slice->serializeState(encoded);
    EXPECT_EQ(encoded.size(), sizeAfterFirst);

    slice->deserializeState(encoded, *bufferManager);
    EXPECT_FALSE(slice->isReduced());

    /// And restoring a slice that is already resident must leave it alone.
    slice->deserializeState(encoded, *bufferManager);
    EXPECT_FALSE(slice->isReduced());
}

/// ---------------------------------------------------------------------------------------------
/// HashMapSlice, via HJSlice. AggregationSlice shares the implementation.
/// ---------------------------------------------------------------------------------------------

TEST_F(ReducibleSliceTest, HashMapSliceRoundTripsAndChainsStillResolve)
{
    HJSlice slice{SLICE_START, SLICE_END, hjSliceArgs(*bufferManager, JoinBuildSideType::Left), NUMBER_OF_WORKER_THREADS};

    /// Hashes drawn from a narrow range against 16 buckets so that entries actually collide; a map of
    /// single-entry chains would say nothing about chain order.
    std::vector<std::vector<ChainSnapshot>> chainsBefore;
    for (uint64_t worker = 0; worker < slice.getNumberOfHashMapsForSide(); ++worker)
    {
        const auto* const buffer
            = slice.getOrCreateHashMapBufferRefForSide(WorkerThreadId(worker), JoinBuildSideType::Left, *bufferManager);
        auto hashMap = ChainedHashMap::load(*buffer);
        for (uint64_t i = 0; i < 60; ++i)
        {
            auto* const entry = static_cast<ChainedHashMapEntry*>(hashMap.insertEntry((i * 7) % 96, bufferManager.get()));
            writePayload(entry, (worker * 1000) + i);
        }
        ASSERT_GT(hashMap.getNumberOfPages(), 1U) << "test is pointless if every map fits on one page";
        chainsBefore.push_back(snapshotChains(hashMap));
    }

    const auto bytesBefore = slice.residentBytes();
    ASSERT_GT(bytesBefore, 0U);

    std::vector<std::byte> encoded;
    slice.serializeState(encoded);
    EXPECT_TRUE(slice.isReduced());
    EXPECT_EQ(slice.residentBytes(), 0U);

    slice.deserializeState(encoded, *bufferManager);
    EXPECT_FALSE(slice.isReduced());
    EXPECT_EQ(slice.residentBytes(), bytesBefore);

    for (uint64_t worker = 0; worker < slice.getNumberOfHashMapsForSide(); ++worker)
    {
        const auto* const buffer = slice.getHashMapBufferRefForSide(WorkerThreadId(worker), JoinBuildSideType::Left);
        ASSERT_NE(buffer, nullptr) << "a slot that held a map before the reduction must hold it again";
        auto hashMap = ChainedHashMap::load(*buffer);
        EXPECT_EQ(hashMap.getTotalNumberOfRecords(), 60U);
        EXPECT_EQ(snapshotChains(hashMap), chainsBefore[worker]);
    }
}

TEST_F(ReducibleSliceTest, HashMapSliceLeavesSlotsNoBuildWorkerEverTouched)
{
    /// Hash map buffers are allocated on first touch, so a slice can be reduced while only some of its
    /// slots exist. The untouched ones must still read back as absent afterwards — not as freshly
    /// allocated empty maps, which is what a slot-state reset would produce.
    HJSlice slice{SLICE_START, SLICE_END, hjSliceArgs(*bufferManager, JoinBuildSideType::Left), NUMBER_OF_WORKER_THREADS};

    const auto* const touched = slice.getOrCreateHashMapBufferRefForSide(WorkerThreadId(0), JoinBuildSideType::Left, *bufferManager);
    auto hashMap = ChainedHashMap::load(*touched);
    for (uint64_t i = 0; i < 10; ++i)
    {
        auto* const entry = static_cast<ChainedHashMapEntry*>(hashMap.insertEntry(i, bufferManager.get()));
        writePayload(entry, i);
    }
    const auto chainsBefore = snapshotChains(hashMap);

    for (uint64_t worker = 1; worker < slice.getNumberOfHashMapsForSide(); ++worker)
    {
        ASSERT_EQ(slice.getHashMapBufferRefForSide(WorkerThreadId(worker), JoinBuildSideType::Left), nullptr);
    }

    std::vector<std::byte> encoded;
    slice.serializeState(encoded);
    slice.deserializeState(encoded, *bufferManager);

    const auto* const restored = slice.getHashMapBufferRefForSide(WorkerThreadId(0), JoinBuildSideType::Left);
    ASSERT_NE(restored, nullptr);
    auto restoredMap = ChainedHashMap::load(*restored);
    EXPECT_EQ(snapshotChains(restoredMap), chainsBefore);

    for (uint64_t worker = 1; worker < slice.getNumberOfHashMapsForSide(); ++worker)
    {
        EXPECT_EQ(slice.getHashMapBufferRefForSide(WorkerThreadId(worker), JoinBuildSideType::Left), nullptr)
            << "slot " << worker << " was never touched and must not come back as an empty map";
    }
}

TEST_F(ReducibleSliceTest, HashMapSliceRoundTripsWithNothingInItAtAll)
{
    /// The whole slice can be reduced before any build worker first-touched a single slot.
    HJSlice slice{SLICE_START, SLICE_END, hjSliceArgs(*bufferManager, JoinBuildSideType::Left), NUMBER_OF_WORKER_THREADS};
    ASSERT_EQ(slice.residentBytes(), 0U);

    std::vector<std::byte> encoded;
    slice.serializeState(encoded);
    slice.deserializeState(encoded, *bufferManager);

    EXPECT_FALSE(slice.isReduced());
    EXPECT_EQ(slice.residentBytes(), 0U);
    for (uint64_t worker = 0; worker < slice.getNumberOfHashMapsForSide(); ++worker)
    {
        EXPECT_EQ(slice.getHashMapBufferRefForSide(WorkerThreadId(worker), JoinBuildSideType::Left), nullptr);
    }
}

TEST_F(ReducibleSliceTest, HashMapSliceKeepsTheTwoJoinSidesApart)
{
    /// HJSlice packs both build sides into one buffer vector, left first. A reduction that reordered the
    /// slots would silently join a stream against itself.
    HJSlice slice{SLICE_START, SLICE_END, hjSliceArgs(*bufferManager, JoinBuildSideType::Left), NUMBER_OF_WORKER_THREADS};

    for (const auto side : {JoinBuildSideType::Left, JoinBuildSideType::Right})
    {
        const auto* const buffer = slice.getOrCreateHashMapBufferRefForSide(WorkerThreadId(0), side, *bufferManager);
        auto hashMap = ChainedHashMap::load(*buffer);
        auto* const entry = static_cast<ChainedHashMapEntry*>(hashMap.insertEntry(1, bufferManager.get()));
        writePayload(entry, side == JoinBuildSideType::Left ? 111 : 222);
    }

    std::vector<std::byte> encoded;
    slice.serializeState(encoded);
    slice.deserializeState(encoded, *bufferManager);

    for (const auto side : {JoinBuildSideType::Left, JoinBuildSideType::Right})
    {
        const auto* const buffer = slice.getHashMapBufferRefForSide(WorkerThreadId(0), side);
        ASSERT_NE(buffer, nullptr);
        auto hashMap = ChainedHashMap::load(*buffer);
        const auto chains = snapshotChains(hashMap);
        const auto expected = side == JoinBuildSideType::Left ? 111U : 222U;
        bool found = false;
        for (const auto& chain : chains)
        {
            for (const auto& [hash, payload] : chain)
            {
                found = found or payload == expected;
            }
        }
        EXPECT_TRUE(found) << "side " << (side == JoinBuildSideType::Left ? "left" : "right") << " lost its entry";
    }
}
}
