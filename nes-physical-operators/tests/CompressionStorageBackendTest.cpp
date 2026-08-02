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
#include <string>
#include <string_view>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <SliceStore/Spill/CompressionStorageBackend.hpp>
#include <SliceStore/Spill/InMemoryStorageBackend.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/Spill/StorageBackendRegistry.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>

namespace NES
{

class CompressionStorageBackendTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("CompressionStorageBackendTest.log", LogLevel::LOG_DEBUG); }

    static SpillObjectKey makeKey(uint64_t sliceEnd)
    {
        return SpillObjectKey{
            .queryId = 7,
            .originId = OriginId{1},
            .sliceEnd = Timestamp{sliceEnd},
            .thread = WorkerThreadId{0},
            .role = SpillRole::NljLeft,
            .index = 0,
        };
    }

    static std::vector<std::byte> bytesOf(std::string_view s)
    {
        std::vector<std::byte> out(s.size());
        std::memcpy(out.data(), s.data(), s.size());
        return out;
    }

    /// Reads the whole object with a deliberately small buffer to exercise partial fills across frames.
    static std::vector<std::byte> readAll(SpillReader& reader, std::size_t bufSize)
    {
        std::vector<std::byte> out;
        std::vector<std::byte> buffer(bufSize);
        while (true)
        {
            auto result = reader.readNext(std::span<std::byte>{buffer.data(), buffer.size()}).get();
            EXPECT_TRUE(result.has_value()) << "readNext failed";
            if (!result.has_value() || result.value() == 0)
            {
                break;
            }
            const auto prev = out.size();
            out.resize(prev + result.value());
            std::memcpy(out.data() + prev, buffer.data(), result.value());
        }
        return out;
    }
};

TEST_F(CompressionStorageBackendTest, RegisteredAsPluginWrappingNamedInnerBackend)
{
    (void)forceLinkCompressionStorageBackend();
    (void)forceLinkInMemoryStorageBackend();
    EXPECT_TRUE(StorageBackendRegistry::instance().contains("compression"));

    auto backend = StorageBackendRegistry::instance().create(
        "compression", StorageBackendArgs{.innerBackendName = "in-memory", .compressionLevel = 3});
    ASSERT_NE(backend, nullptr);
}

TEST_F(CompressionStorageBackendTest, SelfWrapIsRejected)
{
    (void)forceLinkCompressionStorageBackend();
    auto backend = StorageBackendRegistry::instance().create("compression", StorageBackendArgs{.innerBackendName = "compression"});
    EXPECT_EQ(backend, nullptr);
}

TEST_F(CompressionStorageBackendTest, RoundTripAcrossMultipleAppends)
{
    auto inner = std::make_shared<InMemoryStorageBackend>();
    CompressionStorageBackend backend{inner, 3};
    const auto key = makeKey(100);

    const auto first = bytesOf("the quick brown fox jumps over the lazy dog ");
    const auto second = bytesOf(std::string(2048, 'a')); /// highly compressible
    auto writer = backend.openWrite(key);
    ASSERT_TRUE(writer->append(std::span<const std::byte>{first}).get().has_value());
    ASSERT_TRUE(writer->append(std::span<const std::byte>{second}).get().has_value());
    ASSERT_TRUE(writer->append(std::span<const std::byte>{}).get().has_value()); /// empty append == no-op
    ASSERT_TRUE(writer->close().get().has_value());

    auto reader = backend.openRead(key);
    ASSERT_TRUE(reader.has_value());
    const auto out = readAll(*reader.value(), 7); /// odd buffer size spans frame boundaries

    std::vector<std::byte> expected = first;
    expected.insert(expected.end(), second.begin(), second.end());
    ASSERT_EQ(out.size(), expected.size());
    EXPECT_EQ(out, expected);

    /// The compressible payload must actually shrink on the wire (frames are smaller than raw input).
    EXPECT_LT(inner->bytesStored(key), first.size() + second.size());
}

TEST_F(CompressionStorageBackendTest, TruncatedFrameHeaderIsCorrupted)
{
    auto inner = std::make_shared<InMemoryStorageBackend>();
    CompressionStorageBackend backend{inner, 3};
    const auto key = makeKey(200);

    /// Write 3 raw bytes straight to the inner backend — fewer than the 8-byte frame header.
    const auto garbage = bytesOf("xyz");
    auto innerWriter = inner->openWrite(key);
    ASSERT_TRUE(innerWriter->append(std::span<const std::byte>{garbage}).get().has_value());
    ASSERT_TRUE(innerWriter->close().get().has_value());

    auto reader = backend.openRead(key);
    ASSERT_TRUE(reader.has_value());
    std::array<std::byte, 64> buf{};
    auto result = reader.value()->readNext(std::span<std::byte>{buf.data(), buf.size()}).get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, IoErrorCode::Corrupted);
}

/// The compressed tiers move slices between each other with copySpillObject, which is byte-oriented.
/// It only works because a compressed backend's reader yields plaintext and its writer takes plaintext,
/// so the decorator composes in both directions.
TEST_F(CompressionStorageBackendTest, CopyBetweenCompressedAndRawTiersPreservesPlaintext)
{
    auto rawInner = std::make_shared<InMemoryStorageBackend>();
    auto compressedInner = std::make_shared<InMemoryStorageBackend>();
    CompressionStorageBackend compressed{compressedInner, /*level*/ 3};
    const auto key = makeKey(100);
    /// Highly repetitive so the compressed form is unambiguously smaller than the plaintext.
    const auto plaintext = bytesOf(std::string(8192, 'x'));

    auto writer = compressed.openWrite(key);
    ASSERT_TRUE(writer->append(std::span<const std::byte>{plaintext}).get().has_value());
    ASSERT_TRUE(writer->close().get().has_value());
    EXPECT_LT(compressedInner->bytesStored(key), plaintext.size());

    /// Demotion: compressed tier -> raw tier. The raw tier must end up holding the plaintext.
    ASSERT_TRUE(copySpillObject(compressed, *rawInner, key).has_value());
    EXPECT_EQ(rawInner->bytesStored(key), plaintext.size());

    /// And back: raw tier -> compressed tier, then read it out as plaintext again.
    auto roundTripInner = std::make_shared<InMemoryStorageBackend>();
    CompressionStorageBackend roundTrip{roundTripInner, /*level*/ 3};
    ASSERT_TRUE(copySpillObject(*rawInner, roundTrip, key).has_value());
    EXPECT_LT(roundTripInner->bytesStored(key), plaintext.size());

    auto reader = roundTrip.openRead(key);
    ASSERT_TRUE(reader.has_value());
    EXPECT_EQ(readAll(*reader.value(), 64), plaintext);
}

}
