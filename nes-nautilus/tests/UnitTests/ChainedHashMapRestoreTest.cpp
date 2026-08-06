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

/// Covers the one thing BufferTreeCodec cannot do on its own: ChainedHashMap stores its chain heads, its
/// per-entry `next` links and its end sentinel as raw pointers, which do not survive being written to
/// bytes and read back at a different address. rebuildChains() has to recreate all of them, and it has to
/// recreate the chains in their original ORDER, because chain order is probe order and probe order is the
/// row order of a hash join's output.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/Allocator/NesDefaultMemoryAllocator.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/BufferTreeCodec.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <gtest/gtest.h> /// NOLINT(misc-include-cleaner): consumed via macros expanded from rapidcheck/gtest.h

/// Umbrella header: rapidcheck spreads rc::gen and the DefaultArbitrary specialisations over impl headers
/// that are not meant to be included one by one.
/// NOLINTNEXTLINE(misc-include-cleaner)
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

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
/// Small on purpose: entrySize is sizeof(ChainedHashMapEntry) + 16, so 200 bytes holds a handful of
/// entries and the inserts below spill across many pages. A single-page map would leave the page walk in
/// rebuildChains() untested.
constexpr uint64_t PAGE_SIZE = 200;
constexpr uint64_t NUMBER_OF_ENTRIES = 97;

/// One chain, flattened head-first: the (hash, payload) of every entry in link order.
using ChainSnapshot = std::vector<std::pair<uint64_t, uint64_t>>;

std::byte* payloadOf(ChainedHashMapEntry* entry)
{
    /// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<std::byte*>(entry) + sizeof(ChainedHashMapEntry);
}

void writePayload(ChainedHashMapEntry* entry, const uint64_t payload)
{
    std::memcpy(payloadOf(entry), &payload, sizeof(payload));
}

uint64_t readPayload(ChainedHashMapEntry* entry)
{
    uint64_t payload = 0;
    std::memcpy(&payload, payloadOf(entry), sizeof(payload));
    return payload;
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
            chain.emplace_back(entry->hash, readPayload(entry));
        }
        snapshot.push_back(std::move(chain));
    }
    return snapshot;
}

/// Every entry a chain links to must live inside one of this map's own pages. This is what actually
/// proves the links were rebuilt rather than carried over: a pointer left over from the encoded map
/// points at freed memory, which is outside every page here regardless of what that memory now holds.
void expectEntriesLiveInOwnPages(ChainedHashMap& hashMap)
{
    std::vector<std::pair<const std::byte*, const std::byte*>> pageRanges;
    for (uint64_t pageIdx = 0; pageIdx < hashMap.getNumberOfPages(); ++pageIdx)
    {
        const auto page = hashMap.getPage(pageIdx);
        const auto memory = page.getAvailableMemoryArea();
        pageRanges.emplace_back(memory.data(), memory.data() + memory.size());
    }

    uint64_t linkedEntries = 0;
    for (uint64_t pos = 0; pos < hashMap.getNumberOfChains(); ++pos)
    {
        for (auto* entry = hashMap.getChain(pos); entry != nullptr; entry = entry->next)
        {
            /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            const auto* const address = reinterpret_cast<const std::byte*>(entry);
            const bool inSomePage = std::ranges::any_of(
                pageRanges, [address](const auto& range) { return address >= range.first && address < range.second; });
            ASSERT_TRUE(inSomePage) << "chain " << pos << " links to an entry outside this map's pages";
            ++linkedEntries;
        }
    }
    EXPECT_EQ(linkedEntries, hashMap.getTotalNumberOfRecords()) << "every stored entry must be reachable from exactly one chain";
}

/// Hashes chosen so several entries collide into the same chain, which is what gives the ordering
/// assertion teeth: with 64 buckets the mask is 127, so i and i + 128 share a chain.
uint64_t hashFor(const uint64_t i)
{
    return (i * 7) % 384;
}

std::shared_ptr<AbstractBufferProvider> makeBufferManager()
{
    return BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NesDefaultMemoryAllocator>());
}

/// Builds a populated map, snapshots it, encodes it, then destroys every handle to it before returning.
/// The original must be gone: if it were still alive, chains left pointing into it would still resolve
/// and the comparison below would pass on a broken rebuild.
struct EncodedMap
{
    std::vector<std::byte> encoded;
    std::vector<ChainSnapshot> chains;
    uint64_t numberOfPages{0};
};

/// Inserts one entry per given hash, with the entry's index as its payload so that every entry is
/// distinguishable and chain order is observable.
EncodedMap buildAndEncode(AbstractBufferProvider& bufferProvider, const std::span<const uint64_t> hashes)
{
    EncodedMap result;
    const ChainedHashMapConfig config{
        .entrySize = sizeof(ChainedHashMapEntry) + KEY_SIZE + VALUE_SIZE, .numberOfBuckets = NUMBER_OF_BUCKETS, .pageSize = PAGE_SIZE};
    auto mainBuffer = bufferProvider.getUnpooledBuffer(config.bufferSize());
    EXPECT_TRUE(mainBuffer.has_value());
    ChainedHashMap::init(mainBuffer.value(), config);

    auto original = ChainedHashMap::load(mainBuffer.value());
    for (uint64_t i = 0; i < hashes.size(); ++i)
    {
        auto* const entry = dynamic_cast<ChainedHashMapEntry*>(original.insertEntry(hashes[i], &bufferProvider));
        writePayload(entry, i);
    }

    result.chains = snapshotChains(original);
    result.numberOfPages = original.getNumberOfPages();
    BufferTreeCodec::write(result.encoded, mainBuffer.value());
    return result;
}

EncodedMap buildAndEncode(AbstractBufferProvider& bufferProvider, const uint64_t numberOfEntries)
{
    std::vector<uint64_t> hashes;
    hashes.reserve(numberOfEntries);
    for (uint64_t i = 0; i < numberOfEntries; ++i)
    {
        hashes.push_back(hashFor(i));
    }
    return buildAndEncode(bufferProvider, hashes);
}
}

TEST(ChainedHashMapRestoreTest, RebuildChainsReproducesEveryChainInOrder)
{
    auto bufferManager = makeBufferManager();
    const auto source = buildAndEncode(*bufferManager, NUMBER_OF_ENTRIES);
    ASSERT_GT(source.numberOfPages, 1U) << "test is pointless if everything fits on one page";

    std::span<const std::byte> stream{source.encoded};
    auto restoredBuffer = BufferTreeCodec::read(stream, *bufferManager);
    ASSERT_TRUE(stream.empty());

    auto restored = ChainedHashMap::load(restoredBuffer);
    ASSERT_EQ(restored.getTotalNumberOfRecords(), NUMBER_OF_ENTRIES);
    ASSERT_EQ(restored.getNumberOfPages(), source.numberOfPages);

    restored.rebuildChains();

    EXPECT_EQ(snapshotChains(restored), source.chains);
    expectEntriesLiveInOwnPages(restored);
}

TEST(ChainedHashMapRestoreTest, RebuildChainsRepointsTheEndSentinel)
{
    auto bufferManager = makeBufferManager();
    const auto source = buildAndEncode(*bufferManager, NUMBER_OF_ENTRIES);

    std::span<const std::byte> stream{source.encoded};
    auto restoredBuffer = BufferTreeCodec::read(stream, *bufferManager);
    auto restored = ChainedHashMap::load(restoredBuffer);
    restored.rebuildChains();

    /// The sentinel is self-referential, so after a restore it must point inside the NEW main buffer.
    /// Straight off the wire it still points into the encoded map's buffer, which is what this catches.
    const auto memory = restoredBuffer.getAvailableMemoryArea();
    /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto* const sentinel = reinterpret_cast<const std::byte*>(restored.getChain(restored.getNumberOfChains()));
    EXPECT_GE(sentinel, memory.data());
    EXPECT_LT(sentinel, memory.data() + memory.size());
}

TEST(ChainedHashMapRestoreTest, RebuildChainsHandlesAnEmptyMap)
{
    auto bufferManager = makeBufferManager();
    const auto source = buildAndEncode(*bufferManager, 0);
    ASSERT_EQ(source.numberOfPages, 0U);

    std::span<const std::byte> stream{source.encoded};
    auto restoredBuffer = BufferTreeCodec::read(stream, *bufferManager);
    auto restored = ChainedHashMap::load(restoredBuffer);

    /// A slice can be reduced before anything was ever inserted into one of its per-thread maps, so the
    /// no-storage-space path has to be safe rather than tripping the precondition inside getPage().
    restored.rebuildChains();

    EXPECT_EQ(restored.getTotalNumberOfRecords(), 0U);
    for (uint64_t pos = 0; pos < restored.getNumberOfChains(); ++pos)
    {
        EXPECT_EQ(restored.getChain(pos), nullptr);
    }
}

/// The tests above pin individual mechanics against one hand-picked shape. This states the contract they
/// are mechanics of: whatever was inserted comes back, chain for chain, in insertion order. rapidcheck
/// picks the shape, so empty maps, single-entry chains, long collision chains and maps that straddle a
/// page boundary all get covered without anyone having to think of them.
///
/// The encoded map is the ground truth and it is destroyed before the comparison — buildAndEncode drops
/// every handle to it — so a rebuild that merely carried the old pointers over has nothing valid to point
/// at and cannot pass by accident.
RC_GTEST_PROP(ChainedHashMapRestoreTest, RestoreReproducesWhateverWasInserted, ())
{
    /// A narrow hash range against 64 buckets (mask 127) guarantees collisions at realistic sizes, which
    /// is where chain order can actually go wrong. An arbitrary uint64 would mostly produce chains of one.
    const auto hashes = *rc::gen::container<std::vector<uint64_t>>(rc::gen::inRange<uint64_t>(0, 512));

    auto bufferManager = makeBufferManager();
    const auto source = buildAndEncode(*bufferManager, hashes);

    std::span<const std::byte> stream{source.encoded};
    auto restoredBuffer = BufferTreeCodec::read(stream, *bufferManager);
    RC_ASSERT(stream.empty());

    auto restored = ChainedHashMap::load(restoredBuffer);
    restored.rebuildChains();

    RC_ASSERT(restored.getTotalNumberOfRecords() == hashes.size());
    RC_ASSERT(restored.getNumberOfPages() == source.numberOfPages);
    RC_ASSERT(snapshotChains(restored) == source.chains);
}
}
