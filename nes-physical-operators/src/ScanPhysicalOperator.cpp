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


#include <ScanPhysicalOperator.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Interface/BufferRef/TupleBufferRef.hpp>
#include <Interface/Record.hpp>
#include <Interface/RecordBuffer.hpp>
#include <ExecutionContext.hpp>
#include <InputFormatter.hpp>
#include <PhysicalOperator.hpp>
#include <val.hpp>
#include <val_arith.hpp>

namespace NES
{

ScanPhysicalOperator::ScanPhysicalOperator(
    std::shared_ptr<TupleBufferRef> bufferRef, std::vector<Record::RecordFieldIdentifier> projections)
    : bufferRef(std::move(bufferRef))
    , projections(std::move(projections))
    , isRawScan(std::dynamic_pointer_cast<InputFormatter>(this->bufferRef) != nullptr)
{
}

nautilus::val<uint64_t> ScanPhysicalOperator::rawScan(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    auto inputFormatterBufferRef = std::dynamic_pointer_cast<InputFormatter>(this->bufferRef);

    if (not inputFormatterBufferRef->indexBuffer(recordBuffer, executionCtx.pipelineMemoryProvider.arena))
    {
        executionCtx.setOpenReturnState(OpenReturnState::REPEAT);
        return nautilus::val<uint64_t>{0};
    }

    /// call open on all child operators
    openChild(executionCtx, recordBuffer);
    /// A child operator may defer the whole buffer (OpenReturnState::REPEAT); no record must be processed then.
    if (executionCtx.getOpenReturnState() == OpenReturnState::REPEAT)
    {
        return nautilus::val<uint64_t>{0};
    }

    /// process buffer
    const auto executeChildLambda = [this](ExecutionContext& executionCtx, Record& record) { executeChild(executionCtx, record); };
    return inputFormatterBufferRef->readBuffer(executionCtx, recordBuffer, executeChildLambda);
}

nautilus::val<uint64_t> ScanPhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    /// initialize global state variables to keep track of the watermark ts and the origin id
    executionCtx.watermarkTs = recordBuffer.getWatermarkTs();
    executionCtx.minTs = recordBuffer.getMinTs();
    executionCtx.originId = recordBuffer.getOriginId();
    executionCtx.currentTs = recordBuffer.getCreatingTs();
    executionCtx.sequenceNumber = recordBuffer.getSequenceNumber();
    executionCtx.chunkNumber = recordBuffer.getChunkNumber();
    executionCtx.lastChunk = recordBuffer.isLastChunk();

    if (isRawScan)
    {
        return rawScan(executionCtx, recordBuffer);
    }
    /// call open on all child operators
    openChild(executionCtx, recordBuffer);
    /// A child operator may defer the whole buffer (OpenReturnState::REPEAT). No record must be processed
    /// then: the repeated task re-executes the buffer from the start, and partial processing would insert
    /// records twice.
    if (executionCtx.getOpenReturnState() == OpenReturnState::REPEAT)
    {
        return nautilus::val<uint64_t>{0};
    }
    /// iterate over records in buffer
    auto numberOfRecords = recordBuffer.getNumRecords();
    for (nautilus::val<uint64_t> i = uint64_t{0}; i < numberOfRecords; i = i + uint64_t{1})
    {
        auto record = bufferRef->readRecord(projections, recordBuffer, i);
        executeChild(executionCtx, record);
    }
    return numberOfRecords;
}

std::optional<PhysicalOperator> ScanPhysicalOperator::getChild() const
{
    return child;
}

void ScanPhysicalOperator::setChild(PhysicalOperator child)
{
    this->child = std::move(child);
}

}
