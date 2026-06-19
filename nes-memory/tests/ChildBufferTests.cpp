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
#include <ctime>
#include <memory>
#include <random>

#include <Runtime/Allocator/NesDefaultMemoryAllocator.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Util/Logger/Logger.hpp>
#include <gtest/gtest.h>

namespace
{
constexpr uint32_t POOLED_BUFFER_SIZE = 1000;
constexpr uint32_t NUMBER_OF_POOLED_BUFFERS = 1024;
constexpr NES::BufferAlignment BUFFER_ALIGNMENT{64};
constexpr double UNPOOLED_MEMORY_FRACTION = 0.9;
constexpr size_t TOTAL_MEMORY_IN_BYTES = 10 * static_cast<size_t>(NUMBER_OF_POOLED_BUFFERS) * POOLED_BUFFER_SIZE;
}

/// Testing if the buffer size stays the same during a store --> load with small, odd sizes
TEST(ChildBufferTests, StoreAndLoadChildBufferOddSizes)
{
    constexpr std::array<size_t, 6> oddSizes = {1, 3, 7, 13, 63, 127};

    auto bufferManager = NES::BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NES::NesDefaultMemoryAllocator>());
    auto baseBuffer = bufferManager->getBufferBlocking();

    for (size_t i = 0; i < std::size(oddSizes); ++i)
    {
        auto buffer = bufferManager->getUnpooledBuffer(oddSizes[i]);
        ASSERT_TRUE(buffer.has_value()) << "Failed to allocate unpooled buffer of size " << oddSizes[i];

        const auto bufferSizeBeforeStore = buffer.value().getBufferSize();
        const auto bufferIndex = baseBuffer.storeChildBuffer(buffer.value());
        EXPECT_EQ(bufferIndex.getRawIndex(), i);
        EXPECT_EQ(baseBuffer.getNumberOfChildBuffers(), i + 1);

        auto loadedBuffer = baseBuffer.loadChildBuffer(bufferIndex);
        EXPECT_EQ(loadedBuffer.getBufferSize(), bufferSizeBeforeStore);
    }
}

/// Testing if the buffer size stays the same during a store --> load with power-of-2 sizes
TEST(ChildBufferTests, StoreAndLoadChildBufferPowerOfTwoSizes)
{
    constexpr std::array<size_t, 13> powerOfTwoSizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    auto bufferManager = NES::BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NES::NesDefaultMemoryAllocator>());
    auto baseBuffer = bufferManager->getBufferBlocking();

    for (size_t i = 0; i < std::size(powerOfTwoSizes); ++i)
    {
        auto buffer = bufferManager->getUnpooledBuffer(powerOfTwoSizes[i]);
        ASSERT_TRUE(buffer.has_value()) << "Failed to allocate unpooled buffer of size " << powerOfTwoSizes[i];

        const auto bufferSizeBeforeStore = buffer.value().getBufferSize();
        const auto bufferIndex = baseBuffer.storeChildBuffer(buffer.value());
        EXPECT_EQ(bufferIndex.getRawIndex(), i);
        EXPECT_EQ(baseBuffer.getNumberOfChildBuffers(), i + 1);

        auto loadedBuffer = baseBuffer.loadChildBuffer(bufferIndex);
        EXPECT_EQ(loadedBuffer.getBufferSize(), bufferSizeBeforeStore);
    }
}

/// Testing if the buffer size stays the same during a store --> load with random sizes
TEST(ChildBufferTests, StoreAndLoadChildBufferRandomSizes)
{
    constexpr size_t minSize = 10;
    constexpr size_t maxSize = 1024;
    constexpr size_t minNumberOfRandomSizes = 100;
    constexpr size_t maxNumberOfRandomSizes = 1'000;

    /// Getting a "random" seed and logging the seed to be able to rerun the test with the same random values
    const auto seed = static_cast<unsigned>(std::time(nullptr));
    NES_INFO("ChildBufferTests seed: {}", seed);
    std::mt19937 gen{seed};
    std::uniform_int_distribution<size_t> sizeDist{minSize, maxSize};
    std::uniform_int_distribution<size_t> countDist{minNumberOfRandomSizes, maxNumberOfRandomSizes};
    const size_t numberOfRandomSizes = countDist(gen);

    auto bufferManager = NES::BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NES::NesDefaultMemoryAllocator>());
    auto baseBuffer = bufferManager->getBufferBlocking();

    for (size_t i = 0; i < numberOfRandomSizes; ++i)
    {
        const size_t randomSize = sizeDist(gen);
        auto buffer = bufferManager->getUnpooledBuffer(randomSize);
        ASSERT_TRUE(buffer.has_value()) << "Failed to allocate unpooled buffer of size " << randomSize;

        const auto bufferSizeBeforeStore = buffer.value().getBufferSize();
        const auto bufferIndex = baseBuffer.storeChildBuffer(buffer.value());
        EXPECT_EQ(bufferIndex.getRawIndex(), i);
        EXPECT_EQ(baseBuffer.getNumberOfChildBuffers(), i + 1);

        auto loadedBuffer = baseBuffer.loadChildBuffer(bufferIndex);
        EXPECT_EQ(loadedBuffer.getBufferSize(), bufferSizeBeforeStore);
    }
}

/// A detached child's memory must return to the pool once the last holder drops it. Without this,
/// PagedVector::drainPages() can only bound growth -- the drained pages stay pinned to the parent.
TEST(ChildBufferTests, DetachChildBufferReturnsMemoryToThePool)
{
    auto bufferManager = NES::BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NES::NesDefaultMemoryAllocator>());
    auto baseBuffer = bufferManager->getBufferBlocking();
    const auto availableWithOnlyBase = bufferManager->getNumberOfAvailableBuffers();

    auto child = bufferManager->getBufferBlocking();
    const auto childIndex = baseBuffer.storeChildBuffer(child);
    ASSERT_EQ(bufferManager->getNumberOfAvailableBuffers(), availableWithOnlyBase - 1);

    {
        /// The loaded handle holds its own reference, so detaching here must not recycle the buffer yet.
        auto loaded = baseBuffer.loadChildBuffer(childIndex);
        baseBuffer.detachChildBuffer(childIndex);
        EXPECT_EQ(bufferManager->getNumberOfAvailableBuffers(), availableWithOnlyBase - 1);
    }
    EXPECT_EQ(bufferManager->getNumberOfAvailableBuffers(), availableWithOnlyBase);
}

/// Child indices are positional, so a detach must leave a tombstone rather than compact the array --
/// otherwise every index handed out after the detached one would silently shift.
TEST(ChildBufferTests, DetachKeepsLaterChildIndicesValid)
{
    auto bufferManager = NES::BufferManager::create(
        TOTAL_MEMORY_IN_BYTES,
        UNPOOLED_MEMORY_FRACTION,
        BUFFER_ALIGNMENT,
        POOLED_BUFFER_SIZE,
        std::make_shared<NES::NesDefaultMemoryAllocator>());
    auto baseBuffer = bufferManager->getBufferBlocking();

    auto first = bufferManager->getUnpooledBuffer(16);
    auto second = bufferManager->getUnpooledBuffer(32);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    const auto firstIndex = baseBuffer.storeChildBuffer(first.value());
    const auto secondIndex = baseBuffer.storeChildBuffer(second.value());
    const auto secondSize = baseBuffer.loadChildBuffer(secondIndex).getBufferSize();

    baseBuffer.detachChildBuffer(firstIndex);

    EXPECT_EQ(baseBuffer.getNumberOfChildBuffers(), 2U);
    EXPECT_EQ(baseBuffer.loadChildBuffer(secondIndex).getBufferSize(), secondSize);
    /// Detaching twice is a no-op, not a double release.
    baseBuffer.detachChildBuffer(firstIndex);
}
