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

/// clear() is what makes slice recycling possible: it must empty the map completely (no stale entry may
/// ever be visible to the next tenant) while keeping every already-allocated page, and insertEntry must
/// then refill those retained pages instead of appending fresh ones.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

#include <Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/Allocator/NesDefaultMemoryAllocator.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <gtest/gtest.h>

namespace NES
{
namespace
{
constexpr uint32_t POOLED_BUFFER_SIZE = 4096;
constexpr uint32_t NUMBER_OF_POOLED_BUFFERS = 1024;
constexpr BufferAlignment BUFFER_ALIGNMENT{64};
constexpr double UNPOOLED_MEMORY_FRACTION = 0.9;
constexpr size_t TOTAL_MEMORY_IN_BYTES = 10 * static_cast<size_t>(NUMBER_OF_POOLED_BUFFERS) * POOLED_BUFFER_SIZE;

constexpr uint64_t KEY_SIZE = 8;
constexpr uint64_t VALUE_SIZE = 8;
constexpr uint64_t NUMBER_OF_BUCKETS = 64;
/// Small on purpose so the inserts below spill across several pages; a single-page map would leave the
/// page-retention behavior untested.
constexpr uint64_t PAGE_SIZE = 200;
constexpr uint64_t NUMBER_OF_ENTRIES = 97;

std::shared_ptr<AbstractBufferProvider> makeBufferManager()
{
    return BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NesDefaultMemoryAllocator>());
}

TupleBuffer makeInitializedMapBuffer(AbstractBufferProvider& bufferProvider)
{
    const ChainedHashMapConfig config{
        .entrySize = sizeof(ChainedHashMapEntry) + KEY_SIZE + VALUE_SIZE, .numberOfBuckets = NUMBER_OF_BUCKETS, .pageSize = PAGE_SIZE};
    auto mainBuffer = bufferProvider.getUnpooledBuffer(config.bufferSize());
    EXPECT_TRUE(mainBuffer.has_value());
    ChainedHashMap::init(mainBuffer.value(), config);
    return mainBuffer.value();
}

void insertEntries(ChainedHashMap& hashMap, AbstractBufferProvider& bufferProvider, const uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i)
    {
        std::ignore = hashMap.insertEntry(i * 7, &bufferProvider);
    }
}
}

TEST(ChainedHashMapClearTest, ClearEmptiesTheMapAndKeepsPages)
{
    auto bufferManager = makeBufferManager();
    auto mainBuffer = makeInitializedMapBuffer(*bufferManager);
    auto hashMap = ChainedHashMap::load(mainBuffer);
    insertEntries(hashMap, *bufferManager, NUMBER_OF_ENTRIES);
    const auto pagesBeforeClear = hashMap.getNumberOfPages();
    ASSERT_GT(pagesBeforeClear, 1U) << "test is pointless if everything fits on one page";

    hashMap.clear();

    EXPECT_EQ(hashMap.getTotalNumberOfRecords(), 0U);
    EXPECT_EQ(hashMap.getNumberOfPages(), pagesBeforeClear);
    for (uint64_t pos = 0; pos < hashMap.getNumberOfChains(); ++pos)
    {
        EXPECT_EQ(hashMap.getChain(pos), nullptr);
    }
    for (uint64_t pageIdx = 0; pageIdx < pagesBeforeClear; ++pageIdx)
    {
        EXPECT_EQ(hashMap.getPage(pageIdx).getNumberOfTuples(), 0U);
    }
}

TEST(ChainedHashMapClearTest, ReinsertAfterClearReusesRetainedPages)
{
    auto bufferManager = makeBufferManager();
    auto mainBuffer = makeInitializedMapBuffer(*bufferManager);
    auto hashMap = ChainedHashMap::load(mainBuffer);
    insertEntries(hashMap, *bufferManager, NUMBER_OF_ENTRIES);
    const auto pagesBeforeClear = hashMap.getNumberOfPages();

    hashMap.clear();

    /// The very first insert lands on retained page 0; without the high-water-mark check in insertEntry
    /// it would append a redundant page instead.
    std::ignore = hashMap.insertEntry(0, bufferManager.get());
    EXPECT_EQ(hashMap.getNumberOfPages(), pagesBeforeClear);
    EXPECT_EQ(hashMap.getPage(0).getNumberOfTuples(), 1U);

    insertEntries(hashMap, *bufferManager, NUMBER_OF_ENTRIES - 1);
    EXPECT_EQ(hashMap.getTotalNumberOfRecords(), NUMBER_OF_ENTRIES);
    EXPECT_EQ(hashMap.getNumberOfPages(), pagesBeforeClear) << "refilling to the same size must not allocate new pages";
}

TEST(ChainedHashMapClearTest, ClearResetsVarSizedPageFillLevels)
{
    auto bufferManager = makeBufferManager();
    auto mainBuffer = makeInitializedMapBuffer(*bufferManager);
    auto hashMap = ChainedHashMap::load(mainBuffer);
    std::ignore = hashMap.allocateSpaceForVarSized(bufferManager.get(), 100);
    ASSERT_GE(hashMap.getNumberOfVarSizedPages(), 1U);

    hashMap.clear();

    for (uint64_t pageIdx = 0; pageIdx < hashMap.getNumberOfVarSizedPages(); ++pageIdx)
    {
        EXPECT_EQ(hashMap.getVarSizedPage(pageIdx).getNumberOfTuples(), 0U);
    }
}

TEST(ChainedHashMapClearTest, ClearOnAnEmptyMapIsANoOp)
{
    auto bufferManager = makeBufferManager();
    auto mainBuffer = makeInitializedMapBuffer(*bufferManager);
    auto hashMap = ChainedHashMap::load(mainBuffer);

    hashMap.clear();

    EXPECT_EQ(hashMap.getTotalNumberOfRecords(), 0U);
    EXPECT_EQ(hashMap.getNumberOfPages(), 0U);
}
}
