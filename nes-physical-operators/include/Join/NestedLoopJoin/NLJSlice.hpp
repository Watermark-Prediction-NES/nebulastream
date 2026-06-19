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

#include <cstdint>
#include <memory>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Interface/PagedVector/PagedVector.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>

namespace NES
{

/// This class represents a single slice for the NestedLoopJoin. It stores all tuples for the left and right stream.
class NLJSlice final : public Slice
{
public:
    NLJSlice(SliceStart sliceStart, SliceEnd sliceEnd, uint64_t numberOfWorkerThreads);

    /// Returns the number of tuples in this slice on either side.
    [[nodiscard]] uint64_t getNumberOfTuplesLeft() const;
    [[nodiscard]] uint64_t getNumberOfTuplesRight() const;

    /// Returns the pointer to the PagedVector on either side.
    [[nodiscard]] PagedVector* getPagedVectorRefLeft(WorkerThreadId workerThreadId) const;
    [[nodiscard]] PagedVector* getPagedVectorRefRight(WorkerThreadId workerThreadId) const;
    [[nodiscard]] PagedVector* getPagedVectorRef(WorkerThreadId workerThreadId, JoinBuildSideType joinBuildSide) const;

    /// Returns the number of distinct per-worker-thread PagedVectors. After combinePagedVectors() is
    /// called, both sides collapse to 1. Used by the spill subsystem to iterate state on disk.
    [[nodiscard]] uint64_t getNumberOfLeftPagedVectors() const noexcept { return leftPagedVectors.size(); }

    [[nodiscard]] uint64_t getNumberOfRightPagedVectors() const noexcept { return rightPagedVectors.size(); }

    /// Moves all tuples in this slice to the PagedVector at 0th index on both sides.
    void combinePagedVectors();

    /// Drops all pages and restores per-side PagedVector count to the original worker-thread count
    /// (combinePagedVectors may have collapsed both sides to 1). New PagedVector objects are cheap
    /// — they own no pages until insertion.
    void reset(SliceStart newStart, SliceEnd newEnd) override;

private:
    std::vector<std::unique_ptr<PagedVector>> leftPagedVectors;
    std::vector<std::unique_ptr<PagedVector>> rightPagedVectors;
    std::mutex combinePagedVectorsMutex;
    uint64_t originalNumberOfWorkerThreads;
};
}
