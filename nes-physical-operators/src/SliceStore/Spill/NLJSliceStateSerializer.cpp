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

#include <SliceStore/Spill/NLJSliceStateSerializer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <future>
#include <memory>
#include <span>
#include <typeindex>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Interface/PagedVector/PagedVector.hpp>
#include <Join/NestedLoopJoin/NLJSlice.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/SliceStateSerializerRegistry.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>

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

struct PageHeader
{
    uint64_t numberOfTuples;
    uint64_t bufferSize;
};

constexpr std::size_t PageHeaderSize = sizeof(PageHeader);

/// Fingerprint helper. Cheap-and-cheerful: XOR-mixes left/right page counts and total tuples so
/// accidental key reuse across slices/threads is caught at restore time.
uint64_t computeFingerprint(uint64_t leftPages, uint64_t rightPages, uint64_t totalTuples)
{
    return (leftPages * 0x9E3779B97F4A7C15ULL) ^ (rightPages * 0xC2B2AE3D27D4EB4FULL) ^ (totalTuples * 0xBF58476D1CE4E5B9ULL);
}

std::expected<void, IoError> writePage(StorageBackend& backend, const SpillObjectKey& key, const TupleBuffer& page, uint64_t& bytesWritten)
{
    const PageHeader header{.numberOfTuples = page.getNumberOfTuples(), .bufferSize = page.getBufferSize()};
    auto writer = backend.openWrite(key);

    std::span<const std::byte> headerSpan{reinterpret_cast<const std::byte*>(&header), PageHeaderSize};
    auto headerResult = writer->append(headerSpan).get();
    if (!headerResult.has_value())
    {
        return std::unexpected{headerResult.error()};
    }
    auto payloadResult = writer->append(page.getAvailableMemoryArea()).get();
    if (!payloadResult.has_value())
    {
        return std::unexpected{payloadResult.error()};
    }
    auto closeResult = writer->close().get();
    if (!closeResult.has_value())
    {
        return std::unexpected{closeResult.error()};
    }
    bytesWritten += PageHeaderSize + header.bufferSize;
    return {};
}

std::expected<TupleBuffer, IoError> readPage(StorageBackend& backend, const SpillObjectKey& key, AbstractBufferProvider& buffers)
{
    auto readerOrError = backend.openRead(key);
    if (!readerOrError.has_value())
    {
        return std::unexpected{readerOrError.error()};
    }
    auto& reader = *readerOrError.value();

    PageHeader header{};
    {
        auto fut = reader.readNext(std::span<std::byte>{reinterpret_cast<std::byte*>(&header), PageHeaderSize});
        const auto headerRead = fut.get();
        if (!headerRead.has_value())
        {
            return std::unexpected{headerRead.error()};
        }
        if (headerRead.value() != PageHeaderSize)
        {
            return std::unexpected{IoError{IoErrorCode::Corrupted, "short read on page header"}};
        }
    }
    auto bufferOpt = buffers.getUnpooledBuffer(header.bufferSize);
    if (!bufferOpt.has_value())
    {
        return std::unexpected{IoError{IoErrorCode::TransientIo, "buffer-pool exhausted during restore"}};
    }
    auto buffer = std::move(bufferOpt.value());
    std::size_t totalRead = 0;
    while (totalRead < header.bufferSize)
    {
        auto span = buffer.getAvailableMemoryArea<std::byte>().subspan(totalRead);
        auto fut = reader.readNext(span);
        const auto chunkRead = fut.get();
        if (!chunkRead.has_value())
        {
            return std::unexpected{chunkRead.error()};
        }
        if (chunkRead.value() == 0)
        {
            return std::unexpected{IoError{IoErrorCode::Corrupted, "short read on page payload"}};
        }
        totalRead += chunkRead.value();
    }
    buffer.setNumberOfTuples(header.numberOfTuples);
    return buffer;
}

}

std::future<std::expected<SpilledSliceHandle, IoError>>
NLJSliceStateSerializer::spill(Slice& slice, StorageBackend& backend, AbstractBufferProvider& /*buffers*/)
{
    auto& nlj = static_cast<NLJSlice&>(slice);
    SpilledSliceHandle handle{};

    auto spillSide = [&](SpillRole role, uint64_t numVectors, auto getPagedVector) -> std::expected<void, IoError>
    {
        for (uint64_t threadIdx = 0; threadIdx < numVectors; ++threadIdx)
        {
            PagedVector* pv = getPagedVector(WorkerThreadId{static_cast<uint32_t>(threadIdx)});
            if (pv == nullptr)
            {
                continue;
            }
            auto pages = pv->drainPages();
            for (std::size_t pageIdx = 0; pageIdx < pages.size(); ++pageIdx)
            {
                const SpillObjectKey key{
                    .queryId = 0,
                    .originId = INVALID_ORIGIN_ID,
                    .sliceEnd = slice.getSliceEnd(),
                    .thread = WorkerThreadId{static_cast<uint32_t>(threadIdx)},
                    .role = role,
                    .index = static_cast<uint16_t>(pageIdx),
                };
                if (auto writeResult = writePage(backend, key, pages[pageIdx], handle.totalBytes); !writeResult.has_value())
                {
                    /// Failure: restore drained pages so the caller still has the data in memory.
                    pv->adoptPages(std::move(pages));
                    return std::unexpected{writeResult.error()};
                }
                handle.keys.push_back(key);
            }
            /// Successful drain: pages are dropped, freeing buffer-pool slots.
        }
        return {};
    };

    if (auto leftResult
        = spillSide(SpillRole::NljLeft, nlj.getNumberOfLeftPagedVectors(), [&](WorkerThreadId t) { return nlj.getPagedVectorRefLeft(t); });
        !leftResult.has_value())
    {
        return ready<std::expected<SpilledSliceHandle, IoError>>(std::unexpected{leftResult.error()});
    }
    if (auto rightResult = spillSide(
            SpillRole::NljRight, nlj.getNumberOfRightPagedVectors(), [&](WorkerThreadId t) { return nlj.getPagedVectorRefRight(t); });
        !rightResult.has_value())
    {
        return ready<std::expected<SpilledSliceHandle, IoError>>(std::unexpected{rightResult.error()});
    }

    handle.stateFingerprint = computeFingerprint(
        nlj.getNumberOfLeftPagedVectors(), nlj.getNumberOfRightPagedVectors(), nlj.getNumberOfTuplesLeft() + nlj.getNumberOfTuplesRight());
    return ready<std::expected<SpilledSliceHandle, IoError>>(std::move(handle));
}

std::future<std::expected<void, IoError>>
NLJSliceStateSerializer::restore(Slice& slice, const SpilledSliceHandle& handle, StorageBackend& backend, AbstractBufferProvider& buffers)
{
    auto& nlj = static_cast<NLJSlice&>(slice);

    /// Group keys by (role, thread) so we can adopt back into the right PagedVector.
    /// The handle's key ordering preserves page index ordering, so we accumulate then adopt.
    std::vector<std::vector<TupleBuffer>> leftByThread(nlj.getNumberOfLeftPagedVectors());
    std::vector<std::vector<TupleBuffer>> rightByThread(nlj.getNumberOfRightPagedVectors());

    for (const auto& key : handle.keys)
    {
        auto pageOrError = readPage(backend, key, buffers);
        if (!pageOrError.has_value())
        {
            return ready<std::expected<void, IoError>>(std::unexpected{pageOrError.error()});
        }
        const auto threadSlot = key.thread.getRawValue();
        if (key.role == SpillRole::NljLeft && threadSlot < leftByThread.size())
        {
            leftByThread[threadSlot].push_back(std::move(pageOrError.value()));
        }
        else if (key.role == SpillRole::NljRight && threadSlot < rightByThread.size())
        {
            rightByThread[threadSlot].push_back(std::move(pageOrError.value()));
        }
        else
        {
            NES_WARNING("NLJ restore: ignoring page with unexpected role/thread {}:{}", static_cast<uint16_t>(key.role), threadSlot);
        }
    }

    for (uint64_t threadIdx = 0; threadIdx < leftByThread.size(); ++threadIdx)
    {
        if (auto* pv = nlj.getPagedVectorRefLeft(WorkerThreadId{static_cast<uint32_t>(threadIdx)}); pv != nullptr)
        {
            pv->adoptPages(std::move(leftByThread[threadIdx]));
        }
    }
    for (uint64_t threadIdx = 0; threadIdx < rightByThread.size(); ++threadIdx)
    {
        if (auto* pv = nlj.getPagedVectorRefRight(WorkerThreadId{static_cast<uint32_t>(threadIdx)}); pv != nullptr)
        {
            pv->adoptPages(std::move(rightByThread[threadIdx]));
        }
    }
    return ready<std::expected<void, IoError>>({});
}

uint64_t NLJSliceStateSerializer::residentBytes(const Slice& slice) const noexcept
{
    const auto& nlj = static_cast<const NLJSlice&>(slice);
    uint64_t total = 0;
    for (uint64_t threadIdx = 0; threadIdx < nlj.getNumberOfLeftPagedVectors(); ++threadIdx)
    {
        if (const auto* pv = nlj.getPagedVectorRefLeft(WorkerThreadId{static_cast<uint32_t>(threadIdx)}); pv != nullptr)
        {
            total += pv->getNumberOfPages() > 0 ? pv->getNumberOfPages() * pv->getFirstPage().getBufferSize() : 0;
        }
    }
    for (uint64_t threadIdx = 0; threadIdx < nlj.getNumberOfRightPagedVectors(); ++threadIdx)
    {
        if (const auto* pv = nlj.getPagedVectorRefRight(WorkerThreadId{static_cast<uint32_t>(threadIdx)}); pv != nullptr)
        {
            total += pv->getNumberOfPages() > 0 ? pv->getNumberOfPages() * pv->getFirstPage().getBufferSize() : 0;
        }
    }
    return total;
}

namespace
{
const SliceStateSerializerRegistrar registrar{"NLJSlice", std::make_shared<NLJSliceStateSerializer>()};
}

int forceLinkNLJSerializer()
{
    return 1;
}

}
