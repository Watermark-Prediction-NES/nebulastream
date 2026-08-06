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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Interface/PagedVector/PagedVector.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <StateReduction/ReducibleSlice.hpp>

namespace NES
{

struct CreateNewNLJSliceArgs final : CreateNewSlicesArguments
{
    CreateNewNLJSliceArgs(AbstractBufferProvider& bufferProvider, uint64_t tupleSizeLeft, uint64_t tupleSizeRight)
        : bufferProvider(&bufferProvider)
        , tupleSizeLeft(tupleSizeLeft)
        , tupleSizeRight(tupleSizeRight) /// NOLINT(clang-analyzer-optin.cplusplus.UninitializedObject)
    {
    }

    ~CreateNewNLJSliceArgs() override = default;

    AbstractBufferProvider* bufferProvider;
    uint64_t tupleSizeLeft;
    uint64_t tupleSizeRight;
};

/// This class represents a single slice for the NestedLoopJoin. It stores all tuples for the left and right stream.
class NLJSlice final : public Slice, public ReducibleSlice
{
public:
    NLJSlice(
        AbstractBufferProvider& bufferProvider,
        SliceStart sliceStart,
        SliceEnd sliceEnd,
        uint64_t numberOfWorkerThreads,
        uint64_t tupleSizeLeft,
        uint64_t tupleSizeRight);

    /// Returns the number of tuples in this slice on either side.
    [[nodiscard]] uint64_t getNumberOfTuplesLeft() const;
    [[nodiscard]] uint64_t getNumberOfTuplesRight() const;

    /// Returns the pointer to the PagedVector on either side.
    [[nodiscard]] const TupleBuffer* getPagedVectorRefLeft(WorkerThreadId workerThreadId) const;
    [[nodiscard]] const TupleBuffer* getPagedVectorRefRight(WorkerThreadId workerThreadId) const;
    [[nodiscard]] const TupleBuffer* getPagedVectorTupleBufferRef(WorkerThreadId workerThreadId, JoinBuildSideType joinBuildSide) const;

    /// Moves all tuples in this slice to the PagedVector at 0th index on both sides.
    void combinePagedVectors();

    /// ReducibleSlice. The two per-side vectors of PagedVector main buffers are the whole of this
    /// slice's state, and each of them owns its pages and their variable-sized payloads as child
    /// buffers, so releasing them releases everything.
    [[nodiscard]] uint64_t residentBytes() const override;
    void serializeState(std::vector<std::byte>& out) override;
    void deserializeState(std::span<const std::byte> in, AbstractBufferProvider& bufferProvider) override;

private:
    /// ReducibleSlice::stateLock guards these slots against a reduction running while a build or probe
    /// reads them.
    std::vector<TupleBuffer> leftPagedVectorBuffers;
    std::vector<TupleBuffer> rightPagedVectorBuffers;
    std::mutex combinePagedVectorsMutex;
};
}
