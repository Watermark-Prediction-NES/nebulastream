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

#include <SliceStore/Spill/CompressionStorageBackend.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <zstd.h>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/Spill/StorageBackendRegistry.hpp>

namespace NES
{

namespace
{
template <typename T>
std::future<T> ready(T value)
{
    std::promise<T> promise;
    auto future = promise.get_future();
    promise.set_value(std::move(value));
    return future;
}

/// Frame layout: [uint32 compressedLen][uint32 rawLen][compressed bytes].
constexpr std::size_t HeaderBytes = 8;
/// Appended bytes accumulate until they reach this size, then flush as one frame. Frame boundaries are
/// independent of append boundaries (the reader serves across them), so this keeps frames large — one
/// zstd call per ~256 KB instead of one per tiny append (a serializer may append 8-byte hashes). Memory
/// stays bounded (one pending buffer), so we never materialise the whole slice.
constexpr std::size_t FlushThreshold = 256UL * 1024;
}

/// Buffers appended bytes and flushes them as bounded zstd frames to the inner writer. Both shipped
/// inner backends consume the span synchronously, so flushFrame() drains the (already-resolved) inner
/// future while the frame buffer is alive.
class CompressionSpillWriter final : public SpillWriter
{
public:
    CompressionSpillWriter(std::unique_ptr<SpillWriter> inner, int level) : inner(std::move(inner)), level(level) { }

    [[nodiscard]] std::future<std::expected<void, IoError>> append(std::span<const std::byte> bytes) override
    {
        pending.insert(pending.end(), bytes.begin(), bytes.end());
        if (pending.size() >= FlushThreshold)
        {
            if (auto result = flushFrame(); !result.has_value())
            {
                return ready<std::expected<void, IoError>>(std::unexpected{result.error()});
            }
        }
        return ready<std::expected<void, IoError>>({});
    }

    [[nodiscard]] std::future<std::expected<void, IoError>> close() override
    {
        if (!pending.empty())
        {
            if (auto result = flushFrame(); !result.has_value())
            {
                return ready<std::expected<void, IoError>>(std::unexpected{result.error()});
            }
        }
        return inner->close();
    }

private:
    [[nodiscard]] std::expected<void, IoError> flushFrame()
    {
        const std::size_t bound = ZSTD_compressBound(pending.size());
        std::vector<std::byte> frame(HeaderBytes + bound);
        const std::size_t compressedLen = ZSTD_compress(frame.data() + HeaderBytes, bound, pending.data(), pending.size(), level);
        if (ZSTD_isError(compressedLen) != 0U)
        {
            return std::unexpected{IoError{IoErrorCode::Corrupted, ZSTD_getErrorName(compressedLen)}};
        }
        const auto compressed32 = static_cast<uint32_t>(compressedLen);
        const auto raw32 = static_cast<uint32_t>(pending.size());
        std::memcpy(frame.data(), &compressed32, sizeof(compressed32));
        std::memcpy(frame.data() + sizeof(compressed32), &raw32, sizeof(raw32));
        frame.resize(HeaderBytes + compressedLen);
        auto result = inner->append(frame).get();
        pending.clear();
        return result;
    }

    std::unique_ptr<SpillWriter> inner;
    int level;
    std::vector<std::byte> pending;
};

/// Decompresses one frame at a time into `buffer`, serving readNext() from it and refilling on demand.
class CompressionSpillReader final : public SpillReader
{
public:
    explicit CompressionSpillReader(std::unique_ptr<SpillReader> inner) : inner(std::move(inner)) { }

    /// Fills dst greedily across frame boundaries (like the local-file / in-memory readers), so a
    /// serializer read that requires a full fill works even when its bytes span several frames.
    /// Returns a short count only at EOF; 0 means EOF.
    [[nodiscard]] std::future<std::expected<uint64_t, IoError>> readNext(std::span<std::byte> dst) override
    {
        if (dst.empty())
        {
            return ready<std::expected<uint64_t, IoError>>(uint64_t{0});
        }
        uint64_t total = 0;
        while (total < dst.size())
        {
            if (cursor >= buffer.size())
            {
                auto refilled = refill();
                if (!refilled.has_value())
                {
                    return ready<std::expected<uint64_t, IoError>>(std::unexpected{refilled.error()});
                }
                if (!refilled.value())
                {
                    break;
                }
            }
            const auto n = std::min<uint64_t>(buffer.size() - cursor, dst.size() - total);
            std::memcpy(dst.data() + total, buffer.data() + cursor, n);
            cursor += n;
            total += n;
        }
        return ready<std::expected<uint64_t, IoError>>(total);
    }

private:
    /// Reads exactly dst.size() bytes from the inner reader unless EOF is hit first.
    /// Returns the count actually read (< dst.size() only at EOF), or the inner error.
    [[nodiscard]] std::expected<uint64_t, IoError> readFull(std::span<std::byte> dst)
    {
        uint64_t total = 0;
        while (total < dst.size())
        {
            auto chunk = inner->readNext(dst.subspan(total)).get();
            if (!chunk.has_value())
            {
                return std::unexpected{chunk.error()};
            }
            if (chunk.value() == 0)
            {
                break;
            }
            total += chunk.value();
        }
        return total;
    }

    /// true == a frame was decompressed into `buffer`; false == clean EOF; error == truncation/corruption.
    [[nodiscard]] std::expected<bool, IoError> refill()
    {
        std::array<std::byte, HeaderBytes> header{};
        auto headerRead = readFull(header);
        if (!headerRead.has_value())
        {
            return std::unexpected{headerRead.error()};
        }
        if (headerRead.value() == 0)
        {
            return false;
        }
        if (headerRead.value() != HeaderBytes)
        {
            return std::unexpected{IoError{IoErrorCode::Corrupted, "truncated compression frame header"}};
        }
        uint32_t compressedLen = 0;
        uint32_t rawLen = 0;
        std::memcpy(&compressedLen, header.data(), sizeof(compressedLen));
        std::memcpy(&rawLen, header.data() + sizeof(compressedLen), sizeof(rawLen));

        std::vector<std::byte> compressed(compressedLen);
        auto payloadRead = readFull(compressed);
        if (!payloadRead.has_value())
        {
            return std::unexpected{payloadRead.error()};
        }
        if (payloadRead.value() != compressedLen)
        {
            return std::unexpected{IoError{IoErrorCode::Corrupted, "truncated compression frame payload"}};
        }

        buffer.resize(rawLen);
        const std::size_t decompressed = ZSTD_decompress(buffer.data(), rawLen, compressed.data(), compressedLen);
        if (ZSTD_isError(decompressed) != 0U || decompressed != rawLen)
        {
            return std::unexpected{IoError{IoErrorCode::Corrupted, "zstd decompress failed"}};
        }
        cursor = 0;
        return true;
    }

    std::unique_ptr<SpillReader> inner;
    std::vector<std::byte> buffer;
    std::size_t cursor{0};
};

CompressionStorageBackend::CompressionStorageBackend(std::shared_ptr<StorageBackend> inner_, uint32_t level_)
    : inner(std::move(inner_)), level(level_)
{
}

std::unique_ptr<SpillWriter> CompressionStorageBackend::openWrite(const SpillObjectKey& key)
{
    return std::make_unique<CompressionSpillWriter>(inner->openWrite(key), static_cast<int>(level));
}

std::expected<std::unique_ptr<SpillReader>, IoError> CompressionStorageBackend::openRead(const SpillObjectKey& key)
{
    auto innerReader = inner->openRead(key);
    if (!innerReader.has_value())
    {
        return std::unexpected{innerReader.error()};
    }
    return std::make_unique<CompressionSpillReader>(std::move(innerReader.value()));
}

std::future<std::expected<void, IoError>> CompressionStorageBackend::removeAsync(const SpillObjectKey& key)
{
    return inner->removeAsync(key);
}

void CompressionStorageBackend::waitForCompletion(std::optional<SpillObjectKey> key)
{
    inner->waitForCompletion(key);
}

namespace
{
const StorageBackendRegistrar registrar{
    "compression",
    [](const StorageBackendArgs& args) -> std::shared_ptr<StorageBackend>
    {
        std::string innerName = args.innerBackendName;
        std::ranges::transform(innerName, innerName.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (innerName == "COMPRESSION")
        {
            return nullptr;
        }
        auto inner = StorageBackendRegistry::instance().create(args.innerBackendName, args);
        return inner ? std::make_shared<CompressionStorageBackend>(std::move(inner), args.compressionLevel) : nullptr;
    }};
}

int forceLinkCompressionStorageBackend()
{
    return 1;
}

}
