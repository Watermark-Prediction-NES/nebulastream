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
#include <Aggregation/AggregationOperatorHandler.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <Aggregation/AggregationSlice.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Interface/HashMap/HashMap.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <PipelineExecutionContext.hpp>
#include <StateReductionConfiguration.hpp>
#include <WindowBasedOperatorHandler.hpp>

namespace NES
{

AggregationOperatorHandler::AggregationOperatorHandler(
    const std::vector<OriginId>& inputOrigins,
    const OriginId outputOriginId,
    std::unique_ptr<WindowSlicesStoreInterface> sliceAndWindowStore,
    const uint64_t maxNumberOfBuckets,
    const StateReductionConfiguration& stateReductionConfiguration)
    : WindowBasedOperatorHandler(inputOrigins, outputOriginId, std::move(sliceAndWindowStore), stateReductionConfiguration)
    , setupAlreadyCalled(false)
    , rollingAverageNumberOfKeys(RollingAverage<uint64_t>{100})
    , maxNumberOfBuckets(maxNumberOfBuckets)
{
}

SliceCreateFunction AggregationOperatorHandler::getCreateNewSlicesFunction(const CreateNewSlicesArguments& newSlicesArguments) const
{
    PRECONDITION(
        numberOfWorkerThreads > 0, "Number of worker threads not set for window based operator. Was setWorkerThreads() being called?");
    auto newHashMapArgs = dynamic_cast<const CreateNewHashMapSliceArgs&>(newSlicesArguments);
    newHashMapArgs.numberOfBuckets = std::clamp(rollingAverageNumberOfKeys.rlock()->getAverage(), 1UL, maxNumberOfBuckets);
    return std::function(
        [outputOriginId = outputOriginId, numberOfWorkerThreads = numberOfWorkerThreads, copyOfNewHashMapArgs = newHashMapArgs](
            SliceStart sliceStart,
            SliceEnd sliceEnd,
            const std::shared_ptr<Slice>& recycledCandidate) -> std::vector<std::shared_ptr<Slice>>
        {
            if (auto candidate = std::dynamic_pointer_cast<AggregationSlice>(recycledCandidate);
                candidate and candidate->matchesLayout(copyOfNewHashMapArgs, numberOfWorkerThreads, 1))
            {
                candidate->reassign(sliceStart, sliceEnd);
                return {std::move(candidate)};
            }
            NES_TRACE("Creating new aggregation slice with for slice {}-{} for output origin {}", sliceStart, sliceEnd, outputOriginId);
            return {std::make_shared<AggregationSlice>(sliceStart, sliceEnd, copyOfNewHashMapArgs, numberOfWorkerThreads)};
        });
}

void AggregationOperatorHandler::triggerSlices(
    const std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>>& slicesAndWindowInfo,
    PipelineExecutionContext* pipelineCtx)
{
    for (const auto& [windowInfo, allSlices] : slicesAndWindowInfo)
    {
        /// Anything state reduction parked has to be back before the buffers are read, and it stays
        /// pinned from here on. Reducing an emitted slice would not free anything: the buffers collected
        /// below are stored as children of the trigger buffer, which keeps the same memory alive until
        /// the probe is done. The pin is dropped when the slice leaves the store.
        pinSlices(allSlices, *pipelineCtx->getBufferManager());

        /// Getting all hashmaps for each slice that has at least one tuple
        std::vector<TupleBuffer> allHashMapBuffers;
        uint64_t totalNumberOfTuples = 0;
        for (const auto& slice : allSlices)
        {
            const auto aggregationSlice = std::dynamic_pointer_cast<AggregationSlice>(slice);
            for (uint64_t hashMapIdx = 0; hashMapIdx < aggregationSlice->getNumberOfHashMaps(); ++hashMapIdx)
            {
                /// Read-only: a hash map no build worker ever touched for this slice is simply skipped rather than
                /// lazily created here. Trigger-time must never allocate, since that would race with a build worker
                /// concurrently first-touching the same slot (see HashMapSlice::getOrCreateHashMapBufferRef).
                const TupleBuffer* hashMapBuffer = aggregationSlice->getHashMapBufferRefForWorker(WorkerThreadId(hashMapIdx));
                if (hashMapBuffer == nullptr)
                {
                    continue;
                }
                if (const ChainedHashMap hashMap = ChainedHashMap::load(*hashMapBuffer); hashMap.getTotalNumberOfRecords() > 0)
                {
                    /// As the hashmap has one value per key, we can use the number of tuples for the number of keys
                    rollingAverageNumberOfKeys.wlock()->add(hashMap.getTotalNumberOfRecords());
                    allHashMapBuffers.emplace_back(*hashMapBuffer);
                    totalNumberOfTuples += hashMap.getTotalNumberOfRecords();
                }
            }
        }

        /// We need a buffer that is large enough to store an EmittedAggregationWindow
        constexpr auto neededBufferSize = sizeof(EmittedAggregationWindow);
        const auto tupleBufferVal = pipelineCtx->getBufferManager()->getUnpooledBuffer(neededBufferSize);
        if (not tupleBufferVal.has_value())
        {
            throw CannotAllocateBuffer("{}B for the hash join window trigger were requested", neededBufferSize);
        }
        auto tupleBuffer = tupleBufferVal.value();
        /// Store each hash map buffer as a child so the probe can load them via loadChildBuffer(i)
        for (auto hashMapBuffer : allHashMapBuffers)
        {
            std::ignore = tupleBuffer.storeChildBuffer(hashMapBuffer);
        }

        /// It might be that the buffer is not zeroed out.
        std::ranges::fill(tupleBuffer.getAvailableMemoryArea(), std::byte{0});

        /// As we are here "emitting" a buffer, we have to set the originId, the seq number, the watermark and the "number of tuples".
        /// The watermark cannot be the slice end as some buffers might be still waiting to get processed.
        tupleBuffer.setOriginId(outputOriginId);
        tupleBuffer.setSequenceNumber(windowInfo.sequenceNumber);
        tupleBuffer.setChunkNumber(ChunkNumber(ChunkNumber::INITIAL));
        tupleBuffer.setLastChunk(true);
        tupleBuffer.setWatermark(windowInfo.windowInfo.windowStart);
        tupleBuffer.setNumberOfTuples(totalNumberOfTuples);
        tupleBuffer.setCreationTimestampInMS(Timestamp(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()));


        /// Writing all necessary information for the aggregation probe to the buffer via the placement new constructor
        auto tmp = tupleBuffer.getAvailableMemoryArea();
        new (tmp.data()) EmittedAggregationWindow{windowInfo.windowInfo, allHashMapBuffers.size()};


        /// Dispatching the buffer to the probe operator via the task queue.
        pipelineCtx->emitBuffer(tupleBuffer);
        NES_TRACE(
            "Emitted window {}-{} with watermarkTs {} sequenceNumber {} originId {}",
            windowInfo.windowInfo.windowStart,
            windowInfo.windowInfo.windowEnd,
            tupleBuffer.getWatermark(),
            tupleBuffer.getSequenceNumber(),
            tupleBuffer.getOriginId());
    }
}

}
