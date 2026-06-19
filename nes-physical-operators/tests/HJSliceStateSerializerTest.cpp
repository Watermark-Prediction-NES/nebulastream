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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Join/HashJoin/HJSlice.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/BufferManager.hpp>
#include <SliceStore/Spill/HJSliceStateSerializer.hpp>
#include <SliceStore/Spill/InMemoryStorageBackend.hpp>
#include <SliceStore/Spill/SliceStateSerializerRegistry.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <Time/Timestamp.hpp>
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
constexpr uint64_t KeySize = 8;
constexpr uint64_t ValueSize = 8;
constexpr uint64_t PageSize = 4096;
constexpr uint64_t NumberOfBuckets = 64;
constexpr uint64_t NumberOfWorkerThreads = 1;
}

class HJSliceStateSerializerTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite()
    {
        Logger::setupLogging("HJSliceStateSerializerTest.log", LogLevel::LOG_DEBUG);
        /// Pull the registrar TU into the test binary.
        (void)forceLinkHJSerializer();
    }

    void SetUp() override
    {
        Testing::BaseUnitTest::SetUp();
        bufferManager = BufferManager::create(/*bufferSize*/ 4096, /*numOfBuffers*/ 64);
    }

    static CreateNewHashMapSliceArgs makeArgs()
    {
        /// HashMapSlice's destructor requires nautilusCleanup.size() == numberOfInputStreams (= 2 for HJ).
        /// We pass two null shared_ptrs; they are only dereferenced when a hashmap still holds entries at
        /// destruction time. The custom deleter below clears all hashmaps before delete, so the cleanup
        /// callbacks are never invoked in tests.
        std::vector<std::shared_ptr<CreateNewHashMapSliceArgs::NautilusCleanupExec>> nautilusCleanup;
        nautilusCleanup.emplace_back(nullptr);
        nautilusCleanup.emplace_back(nullptr);
        return CreateNewHashMapSliceArgs{std::move(nautilusCleanup), KeySize, ValueSize, PageSize, NumberOfBuckets};
    }

    /// Deleter that clears every hashmap on the slice before invoking delete. Avoids dereferencing
    /// the null cleanup callbacks via the destructor's per-non-empty-hashmap loop.
    struct ClearOnDelete
    {
        void operator()(HJSlice* slice) const noexcept
        {
            if (slice == nullptr)
            {
                return;
            }
            for (uint64_t slot = 0; slot < slice->getNumberOfHashMapsForSide(); ++slot)
            {
                for (auto side : {JoinBuildSideType::Left, JoinBuildSideType::Right})
                {
                    if (auto* hm = static_cast<ChainedHashMap*>(slice->getHashMapPtr(WorkerThreadId{static_cast<uint32_t>(slot)}, side)))
                    {
                        hm->clear();
                    }
                }
            }
            delete slice;
        }
    };

    using SlicePtr = std::unique_ptr<HJSlice, ClearOnDelete>;

    SlicePtr makeSlice()
    {
        return SlicePtr{new HJSlice(SliceStart{Timestamp{0}}, SliceEnd{Timestamp{100}}, makeArgs(), NumberOfWorkerThreads)};
    }

    /// Inserts a synthetic entry whose hash and key are both `keyValue` and whose value is `valueValue`.
    void seedEntry(ChainedHashMap& hashmap, uint64_t keyValue, uint64_t valueValue)
    {
        auto* entry = static_cast<ChainedHashMapEntry*>(hashmap.insertEntry(keyValue, bufferManager.get()));
        auto* payloadPtr = reinterpret_cast<std::byte*>(entry) + sizeof(ChainedHashMapEntry);
        std::memcpy(payloadPtr, &keyValue, sizeof(uint64_t));
        std::memcpy(payloadPtr + sizeof(uint64_t), &valueValue, sizeof(uint64_t));
    }

    /// Walks every chain and collects the keys into a set.
    static std::unordered_set<uint64_t> collectKeySet(const ChainedHashMap& hashmap)
    {
        std::unordered_set<uint64_t> set;
        for (uint64_t chainIdx = 0; chainIdx < hashmap.getNumberOfChains(); ++chainIdx)
        {
            for (auto* entry = hashmap.getStartOfChain(chainIdx); entry != nullptr; entry = entry->next)
            {
                uint64_t key{};
                std::memcpy(&key, reinterpret_cast<const std::byte*>(entry) + sizeof(ChainedHashMapEntry), sizeof(uint64_t));
                set.insert(key);
            }
        }
        return set;
    }

    std::shared_ptr<AbstractBufferProvider> bufferManager;
};

TEST_F(HJSliceStateSerializerTest, RoundTripFixedSizeKeyValueRestoresAllEntries)
{
    auto sourceSlice = makeSlice();
    auto* leftMap = static_cast<ChainedHashMap*>(sourceSlice->getHashMapPtrOrCreate(WorkerThreadId{0}, JoinBuildSideType::Left));
    auto* rightMap = static_cast<ChainedHashMap*>(sourceSlice->getHashMapPtrOrCreate(WorkerThreadId{0}, JoinBuildSideType::Right));

    std::unordered_set<uint64_t> expectedLeft;
    for (uint64_t i = 0; i < 200; ++i)
    {
        seedEntry(*leftMap, i, i * 2);
        expectedLeft.insert(i);
    }
    std::unordered_set<uint64_t> expectedRight;
    for (uint64_t i = 0; i < 50; ++i)
    {
        seedEntry(*rightMap, i + 1000, i * 3);
        expectedRight.insert(i + 1000);
    }

    auto* serializer = SliceStateSerializerRegistry::instance().lookup("HJSlice");
    ASSERT_NE(serializer, nullptr);

    auto backend = std::make_shared<InMemoryStorageBackend>();
    auto spillResult = serializer->spill(*sourceSlice, *backend, *bufferManager).get();
    ASSERT_TRUE(spillResult.has_value());

    const auto handle = std::move(spillResult.value());
    EXPECT_EQ(handle.keys.size(), 2U);
    EXPECT_EQ(backend->numStoredObjects(), 2U);
    EXPECT_EQ(leftMap->getNumberOfTuples(), 0U);
    EXPECT_EQ(rightMap->getNumberOfTuples(), 0U);

    auto restoreSlice = makeSlice();
    auto restoreResult = serializer->restore(*restoreSlice, handle, *backend, *bufferManager).get();
    ASSERT_TRUE(restoreResult.has_value());

    const auto* restoredLeft = static_cast<const ChainedHashMap*>(restoreSlice->getHashMapPtr(WorkerThreadId{0}, JoinBuildSideType::Left));
    const auto* restoredRight
        = static_cast<const ChainedHashMap*>(restoreSlice->getHashMapPtr(WorkerThreadId{0}, JoinBuildSideType::Right));
    ASSERT_NE(restoredLeft, nullptr);
    ASSERT_NE(restoredRight, nullptr);
    EXPECT_EQ(restoredLeft->getNumberOfTuples(), expectedLeft.size());
    EXPECT_EQ(restoredRight->getNumberOfTuples(), expectedRight.size());
    EXPECT_EQ(collectKeySet(*restoredLeft), expectedLeft);
    EXPECT_EQ(collectKeySet(*restoredRight), expectedRight);
}

TEST_F(HJSliceStateSerializerTest, VarSizedHashJoinSpillReturnsTransientIoNotImplemented)
{
    auto slice = makeSlice();
    auto* leftMap = static_cast<ChainedHashMap*>(slice->getHashMapPtrOrCreate(WorkerThreadId{0}, JoinBuildSideType::Left));
    seedEntry(*leftMap, 42, 99);
    /// Drive varSizedSpace allocation by requesting variable-sized storage.
    [[maybe_unused]] auto varSpan = leftMap->allocateSpaceForVarSized(bufferManager.get(), 64);
    EXPECT_GT(leftMap->getNumberOfVarSizedPages(), 0U);

    auto* serializer = SliceStateSerializerRegistry::instance().lookup("HJSlice");
    ASSERT_NE(serializer, nullptr);

    auto backend = std::make_shared<InMemoryStorageBackend>();
    auto spillResult = serializer->spill(*slice, *backend, *bufferManager).get();
    ASSERT_FALSE(spillResult.has_value());
    EXPECT_EQ(spillResult.error().code, IoErrorCode::TransientIo);
    EXPECT_NE(spillResult.error().message.find("var-sized"), std::string::npos);
}

TEST_F(HJSliceStateSerializerTest, EmptyHashMapSpillProducesZeroFiles)
{
    auto slice = makeSlice();
    auto* serializer = SliceStateSerializerRegistry::instance().lookup("HJSlice");
    ASSERT_NE(serializer, nullptr);

    auto backend = std::make_shared<InMemoryStorageBackend>();
    auto spillResult = serializer->spill(*slice, *backend, *bufferManager).get();
    ASSERT_TRUE(spillResult.has_value());

    const auto handle = std::move(spillResult.value());
    EXPECT_TRUE(handle.keys.empty());
    EXPECT_EQ(handle.totalBytes, 0U);
    EXPECT_EQ(backend->numStoredObjects(), 0U);

    auto restoreResult = serializer->restore(*slice, handle, *backend, *bufferManager).get();
    EXPECT_TRUE(restoreResult.has_value());
}

TEST_F(HJSliceStateSerializerTest, HeaderMismatchYieldsCorruptedError)
{
    auto slice = makeSlice();
    auto backend = std::make_shared<InMemoryStorageBackend>();

    /// Hand-write a fake file: 24 bytes of zeros, shorter than the 28-byte header, so restore
    /// fails with a short read on the header.
    const SpillObjectKey fakeKey{
        .queryId = 0,
        .originId = INVALID_ORIGIN_ID,
        .sliceEnd = Timestamp{100},
        .thread = WorkerThreadId{0},
        .role = SpillRole::HashJoinPayload,
        .index = 0,
    };
    auto writer = backend->openWrite(fakeKey);
    std::array<std::byte, 24> zeros{};
    EXPECT_TRUE(writer->append(std::span<const std::byte>{zeros}).get().has_value());
    EXPECT_TRUE(writer->close().get().has_value());

    SpilledSliceHandle handle{};
    handle.keys.push_back(fakeKey);
    handle.totalBytes = zeros.size();

    auto* serializer = SliceStateSerializerRegistry::instance().lookup("HJSlice");
    ASSERT_NE(serializer, nullptr);

    auto restoreResult = serializer->restore(*slice, handle, *backend, *bufferManager).get();
    ASSERT_FALSE(restoreResult.has_value());
    EXPECT_EQ(restoreResult.error().code, IoErrorCode::Corrupted);
}

}
