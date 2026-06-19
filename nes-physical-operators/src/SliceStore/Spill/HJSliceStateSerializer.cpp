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

#include <SliceStore/Spill/HJSliceStateSerializer.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <future>
#include <memory>
#include <ranges>
#include <span>
#include <typeindex>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Join/HashJoin/HJSlice.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/SliceStateSerializerRegistry.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/Logger.hpp>
#include <cpptrace/from_current.hpp>
#include <ErrorHandling.hpp>
#include <HashMapSlice.hpp>

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

/// Explicit constant — 'HJHM' bytes packed little-endian. Multi-char literals like 'HJHM' are
/// implementation-defined; using a hex constant keeps the format stable across compilers.
constexpr uint32_t HJHM_MAGIC = 0x4D484A48U;
constexpr uint16_t HJHM_VERSION = 1;

struct HashMapHeader
{
    uint32_t magic;
    uint16_t version;
    uint16_t reserved;
    uint32_t entrySize;
    uint64_t numberOfChains;
    uint64_t numberOfTuples;
};

constexpr std::size_t HashMapHeaderSize = sizeof(HashMapHeader);

constexpr std::array Sides{JoinBuildSideType::Left, JoinBuildSideType::Right};

uint16_t sideIndex(JoinBuildSideType side) noexcept
{
    return side == JoinBuildSideType::Left ? 0 : 1;
}

uint64_t mixFingerprint(uint64_t current, uint64_t numTuples, uint64_t numChains, uint64_t slot, uint64_t side)
{
    uint64_t result = current;
    result ^= numTuples * 0x9E3779B97F4A7C15ULL;
    result ^= numChains * 0xC2B2AE3D27D4EB4FULL;
    result ^= slot * 0xBF58476D1CE4E5B9ULL;
    result ^= side * 0x94D049BB133111EBULL;
    return result;
}

std::expected<void, IoError>
spillOneHashmap(StorageBackend& backend, const SpillObjectKey& key, ChainedHashMap& hashmap, std::size_t payloadSize, uint64_t& totalBytes)
{
    const HashMapHeader header{
        .magic = HJHM_MAGIC,
        .version = HJHM_VERSION,
        .reserved = 0,
        .entrySize = static_cast<uint32_t>(sizeof(ChainedHashMapEntry) + payloadSize),
        .numberOfChains = hashmap.getNumberOfChains(),
        .numberOfTuples = hashmap.getNumberOfTuples(),
    };

    auto writer = backend.openWrite(key);

    const std::span<const std::byte> headerSpan{reinterpret_cast<const std::byte*>(&header), HashMapHeaderSize};
    if (auto result = writer->append(headerSpan).get(); !result.has_value())
    {
        return std::unexpected{result.error()};
    }

    for (auto chainIdx : std::views::iota(uint64_t{0}, hashmap.getNumberOfChains()))
    {
        for (auto* entry = hashmap.getStartOfChain(chainIdx); entry != nullptr; entry = entry->next)
        {
            const std::span<const std::byte> hashSpan{reinterpret_cast<const std::byte*>(&entry->hash), sizeof(uint64_t)};
            if (auto result = writer->append(hashSpan).get(); !result.has_value())
            {
                return std::unexpected{result.error()};
            }
            const auto* payloadPtr = reinterpret_cast<const std::byte*>(entry) + sizeof(ChainedHashMapEntry);
            const std::span<const std::byte> payloadSpan{payloadPtr, payloadSize};
            if (auto result = writer->append(payloadSpan).get(); !result.has_value())
            {
                return std::unexpected{result.error()};
            }
        }
    }

    if (auto result = writer->close().get(); !result.has_value())
    {
        return std::unexpected{result.error()};
    }

    totalBytes += HashMapHeaderSize + (header.numberOfTuples * (sizeof(uint64_t) + payloadSize));
    return {};
}

std::expected<HashMapHeader, IoError> readHeader(SpillReader& reader)
{
    HashMapHeader header{};
    auto fut = reader.readNext(std::span<std::byte>{reinterpret_cast<std::byte*>(&header), HashMapHeaderSize});
    const auto result = fut.get();
    if (!result.has_value())
    {
        return std::unexpected{result.error()};
    }
    if (result.value() != HashMapHeaderSize)
    {
        return std::unexpected{IoError{IoErrorCode::Corrupted, "HJ restore: short read on header"}};
    }
    return header;
}

std::expected<HashMapHeader, IoError> validateHeader(HashMapHeader header, std::size_t expectedEntrySize)
{
    if (header.magic != HJHM_MAGIC)
    {
        return std::unexpected{IoError{IoErrorCode::Corrupted, std::format("HJ restore: bad magic 0x{:x}", header.magic)}};
    }
    if (header.version != HJHM_VERSION)
    {
        return std::unexpected{IoError{IoErrorCode::Corrupted, std::format("HJ restore: unsupported version {}", header.version)}};
    }
    if (header.entrySize != expectedEntrySize)
    {
        return std::unexpected{IoError{
            IoErrorCode::Corrupted,
            std::format("HJ restore: entrySize mismatch (file={} expected={})", header.entrySize, expectedEntrySize)}};
    }
    return header;
}

std::expected<void, IoError> restoreOneHashmap(
    StorageBackend& backend,
    const SpillObjectKey& key,
    ChainedHashMap& hashmap,
    std::size_t expectedEntrySize,
    std::size_t payloadSize,
    AbstractBufferProvider& buffers)
{
    auto readerOrError = backend.openRead(key);
    if (!readerOrError.has_value())
    {
        return std::unexpected{readerOrError.error()};
    }
    auto& reader = *readerOrError.value();

    const auto headerOrError = readHeader(reader).and_then([&](HashMapHeader h) { return validateHeader(h, expectedEntrySize); });
    if (!headerOrError.has_value())
    {
        return std::unexpected{headerOrError.error()};
    }
    const auto header = headerOrError.value();

    std::vector<std::byte> payloadBuffer(payloadSize);
    CPPTRACE_TRY
    {
        for (auto entryIdx : std::views::iota(uint64_t{0}, header.numberOfTuples))
        {
            uint64_t hash{};
            {
                auto hashRead = reader.readNext(std::span<std::byte>{reinterpret_cast<std::byte*>(&hash), sizeof(uint64_t)}).get();
                if (!hashRead.has_value())
                {
                    return std::unexpected{hashRead.error()};
                }
                if (hashRead.value() != sizeof(uint64_t))
                {
                    return std::unexpected{
                        IoError{IoErrorCode::Corrupted, std::format("HJ restore: short read on hash at entry {}", entryIdx)}};
                }
            }
            std::size_t payloadRead = 0;
            while (payloadRead < payloadSize)
            {
                auto chunkRead = reader.readNext(std::span<std::byte>{payloadBuffer.data() + payloadRead, payloadSize - payloadRead}).get();
                if (!chunkRead.has_value())
                {
                    return std::unexpected{chunkRead.error()};
                }
                if (chunkRead.value() == 0)
                {
                    return std::unexpected{
                        IoError{IoErrorCode::Corrupted, std::format("HJ restore: short read on payload at entry {}", entryIdx)}};
                }
                payloadRead += chunkRead.value();
            }
            auto* entry = static_cast<ChainedHashMapEntry*>(hashmap.insertEntry(hash, &buffers));
            std::memcpy(reinterpret_cast<std::byte*>(entry) + sizeof(ChainedHashMapEntry), payloadBuffer.data(), payloadSize);
        }
    }
    CPPTRACE_CATCH(const Exception& ex)
    {
        return std::unexpected{IoError{IoErrorCode::TransientIo, std::format("HJ restore: buffer pool exhausted: {}", ex.what())}};
    }

    return {};
}

}

std::future<std::expected<SpilledSliceHandle, IoError>>
HJSliceStateSerializer::spill(Slice& slice, StorageBackend& backend, AbstractBufferProvider& /*buffers*/)
{
    auto& hj = static_cast<HJSlice&>(slice);
    const auto& args = hj.getCreateArgs();
    const std::size_t payloadSize = args.keySize + args.valueSize;

    SpilledSliceHandle handle{};

    for (auto slot : std::views::iota(uint64_t{0}, hj.getNumberOfHashMapsForSide()))
    {
        for (auto side : Sides)
        {
            auto* hashmap = static_cast<ChainedHashMap*>(hj.getHashMapPtr(WorkerThreadId{static_cast<uint32_t>(slot)}, side));
            if (hashmap == nullptr)
            {
                continue;
            }
            if (hashmap->getNumberOfVarSizedPages() > 0)
            {
                return ready<std::expected<SpilledSliceHandle, IoError>>(
                    std::unexpected{IoError{IoErrorCode::TransientIo, "var-sized HJ spill not yet implemented"}});
            }
            if (hashmap->getNumberOfTuples() == 0)
            {
                continue;
            }
            const SpillObjectKey key{
                .queryId = 0,
                .originId = INVALID_ORIGIN_ID,
                .sliceEnd = slice.getSliceEnd(),
                .thread = WorkerThreadId{static_cast<uint32_t>(slot)},
                .role = SpillRole::HashJoinPayload,
                .index = sideIndex(side),
            };
            if (auto result = spillOneHashmap(backend, key, *hashmap, payloadSize, handle.totalBytes); !result.has_value())
            {
                return ready<std::expected<SpilledSliceHandle, IoError>>(std::unexpected{result.error()});
            }
            handle.keys.push_back(key);
            handle.stateFingerprint = mixFingerprint(
                handle.stateFingerprint, hashmap->getNumberOfTuples(), hashmap->getNumberOfChains(), slot, sideIndex(side));
            /// Drain after successful write.
            hashmap->clear();
        }
    }

    return ready<std::expected<SpilledSliceHandle, IoError>>(std::move(handle));
}

std::future<std::expected<void, IoError>>
HJSliceStateSerializer::restore(Slice& slice, const SpilledSliceHandle& handle, StorageBackend& backend, AbstractBufferProvider& buffers)
{
    auto& hj = static_cast<HJSlice&>(slice);
    const auto& args = hj.getCreateArgs();
    const std::size_t payloadSize = args.keySize + args.valueSize;
    const std::size_t expectedEntrySize = sizeof(ChainedHashMapEntry) + payloadSize;

    for (const auto& key : handle.keys)
    {
        const auto side = key.index == 0 ? JoinBuildSideType::Left : JoinBuildSideType::Right;
        auto* hashmap = static_cast<ChainedHashMap*>(hj.getHashMapPtrOrCreate(key.thread, side));
        if (hashmap == nullptr)
        {
            return ready<std::expected<void, IoError>>(
                std::unexpected{IoError{IoErrorCode::TransientIo, "HJ restore: getHashMapPtrOrCreate returned null"}});
        }
        if (auto result = restoreOneHashmap(backend, key, *hashmap, expectedEntrySize, payloadSize, buffers); !result.has_value())
        {
            return ready<std::expected<void, IoError>>(std::unexpected{result.error()});
        }
    }

    return ready<std::expected<void, IoError>>({});
}

uint64_t HJSliceStateSerializer::residentBytes(const Slice& slice) const noexcept
{
    const auto& hj = static_cast<const HJSlice&>(slice);
    const auto& args = hj.getCreateArgs();
    uint64_t total = 0;
    for (auto slot : std::views::iota(uint64_t{0}, hj.getNumberOfHashMapsForSide()))
    {
        for (auto side : Sides)
        {
            const auto* hashmap = static_cast<const ChainedHashMap*>(hj.getHashMapPtr(WorkerThreadId{static_cast<uint32_t>(slot)}, side));
            if (hashmap == nullptr)
            {
                continue;
            }
            total += hashmap->getNumberOfPages() * args.pageSize;
            if (hashmap->getNumberOfTuples() > 0)
            {
                total += (hashmap->getNumberOfChains() + 1) * sizeof(void*);
            }
        }
    }
    return total;
}

namespace
{
const SliceStateSerializerRegistrar registrar{"HJSlice", std::make_shared<HJSliceStateSerializer>()};
}

int forceLinkHJSerializer()
{
    return 1;
}

}
