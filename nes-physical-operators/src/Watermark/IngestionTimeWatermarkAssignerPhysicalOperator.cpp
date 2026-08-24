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

#include <Watermark/IngestionTimeWatermarkAssignerPhysicalOperator.hpp>

#include <cstdint>
#include <optional>
#include <utility>
#include <Interface/Record.hpp>
#include <Interface/RecordBuffer.hpp>
#include <Watermark/TimeFunction.hpp>
#include <ExecutionContext.hpp>
#include <PhysicalOperator.hpp>
#include <val_arith.hpp>

namespace NES
{

IngestionTimeWatermarkAssignerPhysicalOperator::IngestionTimeWatermarkAssignerPhysicalOperator(
    IngestionTimeFunction timeFunction, const bool trackMinTs)
    : timeFunction(std::move(timeFunction)), trackMinTs(trackMinTs) { };

nautilus::val<uint64_t>
IngestionTimeWatermarkAssignerPhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    auto numberOfProcessedTuples = openChild(executionCtx, recordBuffer);
    timeFunction.open(executionCtx, recordBuffer);
    auto emptyRecord = Record();
    const auto tsField = [this](ExecutionContext& executionCtx)
    {
        auto emptyRecord = Record();
        return timeFunction.getTs(executionCtx, emptyRecord);
    }(executionCtx);
    if (const auto currentWatermark = executionCtx.watermarkTs; tsField > currentWatermark)
    {
        executionCtx.watermarkTs = tsField;
    }
    if (trackMinTs)
    {
        /// With ingestion time every record in the buffer carries the same timestamp, so min == max.
        executionCtx.minTs = tsField;
    }
    return numberOfProcessedTuples;
}

void IngestionTimeWatermarkAssignerPhysicalOperator::execute(ExecutionContext& executionCtx, Record& record) const
{
    executeChild(executionCtx, record);
}

std::optional<PhysicalOperator> IngestionTimeWatermarkAssignerPhysicalOperator::getChild() const
{
    return child;
}

void IngestionTimeWatermarkAssignerPhysicalOperator::setChild(PhysicalOperator child)
{
    this->child = std::move(child);
}

}
