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
#include <span>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <SliceStore/Spill/InMemoryStorageBackend.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>

namespace NES
{

class StorageBackendContractTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("StorageBackendContractTest.log", LogLevel::LOG_DEBUG); }

    static SpillObjectKey makeKey(uint64_t sliceEnd, SpillRole role = SpillRole::NljLeft)
    {
        return SpillObjectKey{
            .queryId = 7,
            .originId = OriginId{1},
            .sliceEnd = Timestamp{sliceEnd},
            .thread = WorkerThreadId{0},
            .role = role,
            .index = 0,
        };
    }

    static std::vector<std::byte> readAll(SpillReader& reader)
    {
        std::vector<std::byte> out;
        std::array<std::byte, 64> buffer{};
        while (true)
        {
            auto fut = reader.readNext(std::span<std::byte>{buffer.data(), buffer.size()});
            const auto result = fut.get();
            EXPECT_TRUE(result.has_value()) << "readNext failed";
            if (!result.has_value())
            {
                return out;
            }
            const auto n = result.value();
            if (n == 0)
            {
                break;
            }
            const auto prevSize = out.size();
            out.resize(prevSize + n);
            std::memcpy(out.data() + prevSize, buffer.data(), n);
        }
        return out;
    }
};

TEST_F(StorageBackendContractTest, WriteThenReadRoundTrip)
{
    InMemoryStorageBackend backend;
    const auto key = makeKey(100);

    auto writer = backend.openWrite(key);
    const std::array<std::byte, 5> firstChunk{std::byte{'h'}, std::byte{'e'}, std::byte{'l'}, std::byte{'l'}, std::byte{'o'}};
    const std::array<std::byte, 6> secondChunk{
        std::byte{','}, std::byte{' '}, std::byte{'N'}, std::byte{'E'}, std::byte{'S'}, std::byte{'!'}};
    ASSERT_TRUE(writer->append(std::span<const std::byte>{firstChunk}).get().has_value());
    ASSERT_TRUE(writer->append(std::span<const std::byte>{secondChunk}).get().has_value());
    ASSERT_TRUE(writer->close().get().has_value());

    auto readerOrError = backend.openRead(key);
    ASSERT_TRUE(readerOrError.has_value());
    const auto bytes = readAll(*readerOrError.value());
    ASSERT_EQ(bytes.size(), firstChunk.size() + secondChunk.size());
    const std::string asString(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    EXPECT_EQ(asString, "hello, NES!");
}

TEST_F(StorageBackendContractTest, OpenReadMissingKeyReturnsNotFound)
{
    InMemoryStorageBackend backend;
    const auto result = backend.openRead(makeKey(404));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IoErrorCode::NotFound);
}

TEST_F(StorageBackendContractTest, RemoveDeletesObject)
{
    InMemoryStorageBackend backend;
    const auto key = makeKey(200);

    auto writer = backend.openWrite(key);
    const std::array<std::byte, 3> payload{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    ASSERT_TRUE(writer->append(std::span<const std::byte>{payload}).get().has_value());
    ASSERT_TRUE(writer->close().get().has_value());
    EXPECT_EQ(backend.bytesStored(key), payload.size());

    ASSERT_TRUE(backend.removeAsync(key).get().has_value());
    EXPECT_EQ(backend.bytesStored(key), 0U);
    EXPECT_FALSE(backend.openRead(key).has_value());
}

TEST_F(StorageBackendContractTest, WaitForCompletionDoesNotThrow)
{
    InMemoryStorageBackend backend;
    backend.waitForCompletion(std::nullopt);
    backend.waitForCompletion(makeKey(0));
}

TEST_F(StorageBackendContractTest, ReadNextReturnsZeroOnEof)
{
    InMemoryStorageBackend backend;
    const auto key = makeKey(300);
    auto writer = backend.openWrite(key);
    const std::array<std::byte, 2> payload{std::byte{1}, std::byte{2}};
    ASSERT_TRUE(writer->append(std::span<const std::byte>{payload}).get().has_value());
    ASSERT_TRUE(writer->close().get().has_value());

    auto reader = backend.openRead(key);
    ASSERT_TRUE(reader.has_value());
    std::array<std::byte, 32> buf{};
    auto first = reader.value()->readNext(std::span<std::byte>{buf.data(), buf.size()}).get();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first.value(), 2U);
    auto eof = reader.value()->readNext(std::span<std::byte>{buf.data(), buf.size()}).get();
    ASSERT_TRUE(eof.has_value());
    EXPECT_EQ(eof.value(), 0U);
}

TEST_F(StorageBackendContractTest, KeysWithDifferentRolesAreIndependent)
{
    InMemoryStorageBackend backend;
    const auto leftKey = makeKey(100, SpillRole::NljLeft);
    const auto rightKey = makeKey(100, SpillRole::NljRight);

    auto leftWriter = backend.openWrite(leftKey);
    auto rightWriter = backend.openWrite(rightKey);
    const std::array<std::byte, 1> leftBytes{std::byte{'L'}};
    const std::array<std::byte, 1> rightBytes{std::byte{'R'}};
    ASSERT_TRUE(leftWriter->append(std::span<const std::byte>{leftBytes}).get().has_value());
    ASSERT_TRUE(rightWriter->append(std::span<const std::byte>{rightBytes}).get().has_value());
    ASSERT_TRUE(leftWriter->close().get().has_value());
    ASSERT_TRUE(rightWriter->close().get().has_value());

    EXPECT_EQ(backend.numStoredObjects(), 2U);
    EXPECT_EQ(backend.bytesStored(leftKey), 1U);
    EXPECT_EQ(backend.bytesStored(rightKey), 1U);
}

}
