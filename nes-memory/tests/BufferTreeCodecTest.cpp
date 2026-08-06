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
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/Allocator/NesDefaultMemoryAllocator.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/BufferTreeCodec.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <ErrorHandling.hpp>

namespace
{
constexpr uint32_t POOLED_BUFFER_SIZE = 4096;
constexpr uint32_t NUMBER_OF_POOLED_BUFFERS = 1024;
constexpr NES::BufferAlignment BUFFER_ALIGNMENT{64};
constexpr double UNPOOLED_MEMORY_FRACTION = 0.9;
constexpr size_t TOTAL_MEMORY_IN_BYTES = 10 * static_cast<size_t>(NUMBER_OF_POOLED_BUFFERS) * POOLED_BUFFER_SIZE;

std::shared_ptr<NES::AbstractBufferProvider> makeBufferManager()
{
    return NES::BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NES::NesDefaultMemoryAllocator>());
}

/// Fills the buffer's payload with a deterministic pattern derived from `seed`, so a mismatch after a
/// round trip points at which node went wrong rather than just "some bytes differ".
void paint(NES::TupleBuffer& buffer, const uint64_t seed)
{
    auto memory = buffer.getAvailableMemoryArea();
    for (size_t i = 0; i < memory.size(); ++i)
    {
        memory[i] = static_cast<std::byte>((seed * 31 + i * 7) & 0xFF);
    }
    buffer.setNumberOfTuples(seed);
}

/// Recursively compares payload bytes, numberOfTuples and child counts.
void expectTreesEqual(const NES::TupleBuffer& expected, const NES::TupleBuffer& actual, const std::string& path)
{
    const auto expectedMemory = expected.getAvailableMemoryArea();
    const auto actualMemory = actual.getAvailableMemoryArea();
    ASSERT_EQ(expectedMemory.size(), actualMemory.size()) << "payload size differs at " << path;
    ASSERT_EQ(expected.getNumberOfTuples(), actual.getNumberOfTuples()) << "numberOfTuples differs at " << path;
    ASSERT_EQ(expected.getNumberOfChildBuffers(), actual.getNumberOfChildBuffers()) << "child count differs at " << path;

    for (size_t i = 0; i < expectedMemory.size(); ++i)
    {
        ASSERT_EQ(expectedMemory[i], actualMemory[i]) << "payload byte " << i << " differs at " << path;
    }

    for (uint32_t childIdx = 0; childIdx < expected.getNumberOfChildBuffers(); ++childIdx)
    {
        const NES::ChildBufferIndex idx{childIdx};
        expectTreesEqual(expected.loadChildBuffer(idx), actual.loadChildBuffer(idx), path + "/" + std::to_string(childIdx));
    }
}

/// Builds a tree of the given depth where every node has `fanout` children, sizes varying per node.
NES::TupleBuffer buildTree(NES::AbstractBufferProvider& provider, const uint32_t depth, const uint32_t fanout, uint64_t& nodeCounter)
{
    const uint64_t seed = ++nodeCounter;
    /// Deliberately odd, non-power-of-two sizes: the codec must not assume the pool's buffer size.
    const size_t size = 17 + ((seed * 53) % 900);
    auto buffer = provider.getUnpooledBuffer(size);
    EXPECT_TRUE(buffer.has_value());
    paint(buffer.value(), seed);

    if (depth > 0)
    {
        for (uint32_t i = 0; i < fanout; ++i)
        {
            auto child = buildTree(provider, depth - 1, fanout, nodeCounter);
            std::ignore = buffer->storeChildBuffer(child);
        }
    }
    return std::move(buffer.value());
}
}

namespace NES
{
TEST(BufferTreeCodecTest, RoundTripsASingleChildlessBuffer)
{
    auto bufferManager = makeBufferManager();
    auto original = bufferManager->getUnpooledBuffer(123);
    ASSERT_TRUE(original.has_value());
    paint(original.value(), 42);

    std::vector<std::byte> encoded;
    NES::BufferTreeCodec::write(encoded, original.value());

    std::span<const std::byte> stream{encoded};
    auto restored = NES::BufferTreeCodec::read(stream, *bufferManager);

    EXPECT_TRUE(stream.empty()) << "read() must consume exactly what write() produced";
    expectTreesEqual(original.value(), restored, "root");
}

TEST(BufferTreeCodecTest, RoundTripsANestedTreeAndPreservesChildIndices)
{
    auto bufferManager = makeBufferManager();
    uint64_t nodeCounter = 0;
    /// Depth 3, fanout 3 == 40 nodes. Deep enough to cover the three-level main -> page -> varsized
    /// shape that PagedVector produces, and the main -> storageSpace -> page shape ChainedHashMap does.
    auto original = buildTree(*bufferManager, 3, 3, nodeCounter);

    std::vector<std::byte> encoded;
    NES::BufferTreeCodec::write(encoded, original);
    EXPECT_EQ(encoded.size(), NES::BufferTreeCodec::encodedSize(original));

    std::span<const std::byte> stream{encoded};
    auto restored = NES::BufferTreeCodec::read(stream, *bufferManager);

    EXPECT_TRUE(stream.empty());
    /// expectTreesEqual walks by ChildBufferIndex, so passing it proves the indices were reproduced:
    /// a reordered child would compare against the wrong node and mismatch on payload.
    expectTreesEqual(original, restored, "root");
}

TEST(BufferTreeCodecTest, RoundTripsAnUnbalancedTree)
{
    auto bufferManager = makeBufferManager();
    auto root = bufferManager->getUnpooledBuffer(64);
    ASSERT_TRUE(root.has_value());
    paint(root.value(), 1);

    /// One fat child with grandchildren, one leaf, one empty-ish child. Mirrors a PagedVector where only
    /// some pages carry varsized payloads.
    auto fat = bufferManager->getUnpooledBuffer(256);
    ASSERT_TRUE(fat.has_value());
    paint(fat.value(), 2);
    for (uint64_t i = 0; i < 4; ++i)
    {
        auto grandchild = bufferManager->getUnpooledBuffer(32 + i);
        ASSERT_TRUE(grandchild.has_value());
        paint(grandchild.value(), 100 + i);
        std::ignore = fat->storeChildBuffer(grandchild.value());
    }
    auto leaf = bufferManager->getUnpooledBuffer(8);
    ASSERT_TRUE(leaf.has_value());
    paint(leaf.value(), 3);
    auto tiny = bufferManager->getUnpooledBuffer(1);
    ASSERT_TRUE(tiny.has_value());
    paint(tiny.value(), 4);

    std::ignore = root->storeChildBuffer(fat.value());
    std::ignore = root->storeChildBuffer(leaf.value());
    std::ignore = root->storeChildBuffer(tiny.value());

    std::vector<std::byte> encoded;
    NES::BufferTreeCodec::write(encoded, root.value());

    std::span<const std::byte> stream{encoded};
    auto restored = NES::BufferTreeCodec::read(stream, *bufferManager);

    EXPECT_TRUE(stream.empty());
    expectTreesEqual(root.value(), restored, "root");
}

TEST(BufferTreeCodecTest, ConcatenatedTreesAreReadBackInOrder)
{
    /// A slice encodes one tree per worker-thread slot into a single stream, so read() must leave the
    /// span positioned exactly at the next tree.
    auto bufferManager = makeBufferManager();
    uint64_t nodeCounter = 0;
    auto first = buildTree(*bufferManager, 1, 2, nodeCounter);
    auto second = buildTree(*bufferManager, 2, 2, nodeCounter);

    std::vector<std::byte> encoded;
    NES::BufferTreeCodec::write(encoded, first);
    NES::BufferTreeCodec::write(encoded, second);

    std::span<const std::byte> stream{encoded};
    auto restoredFirst = NES::BufferTreeCodec::read(stream, *bufferManager);
    auto restoredSecond = NES::BufferTreeCodec::read(stream, *bufferManager);

    EXPECT_TRUE(stream.empty());
    expectTreesEqual(first, restoredFirst, "first");
    expectTreesEqual(second, restoredSecond, "second");
}

TEST(BufferTreeCodecTest, TruncatedStreamThrows)
{
    auto bufferManager = makeBufferManager();
    uint64_t nodeCounter = 0;
    auto original = buildTree(*bufferManager, 2, 2, nodeCounter);

    std::vector<std::byte> encoded;
    NES::BufferTreeCodec::write(encoded, original);
    ASSERT_GT(encoded.size(), 16U);

    /// Cut mid-tree: the outer node decodes, then a child runs out of bytes.
    encoded.resize(encoded.size() / 2);
    std::span<const std::byte> stream{encoded};
    ASSERT_EXCEPTION_ERRORCODE(std::ignore = NES::BufferTreeCodec::read(stream, *bufferManager), NES::ErrorCode::CannotDeserialize);
}
}
