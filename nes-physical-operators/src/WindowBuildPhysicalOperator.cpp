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
#include <WindowBuildPhysicalOperator.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <Identifiers/Identifiers.hpp>
#include <Interface/RecordBuffer.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <SliceStore/SliceStoreRef.hpp>
#include <Time/Timestamp.hpp>
#include <Watermark/TimeFunction.hpp>
#include <CompilationContext.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <PhysicalOperator.hpp>
#include <WindowBasedOperatorHandler.hpp>
#include <function.hpp>

namespace NES
{

/// Updates the sliceState of all slices and emits buffers, if the slices can be emitted
void checkWindowsTriggerProxy(
    OperatorHandler* ptrOpHandler,
    PipelineExecutionContext* pipelineCtx,
    const Timestamp watermarkTs,
    const SequenceNumber sequenceNumber,
    const ChunkNumber chunkNumber,
    const bool lastChunk,
    const OriginId originId)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    PRECONDITION(pipelineCtx != nullptr, "pipeline context should not be null");

    auto* opHandler = dynamic_cast<WindowBasedOperatorHandler*>(ptrOpHandler);
    const BufferMetaData bufferMetaData(watermarkTs, SequenceData(sequenceNumber, chunkNumber, lastChunk), originId);
    opHandler->checkAndTriggerWindows(bufferMetaData, pipelineCtx);
}

/// Build-path bracket, one call per TupleBuffer. Marks this worker thread as writing slice state so a
/// concurrent GC tick cannot drain a still-filling slice underneath an in-flight insert.
void enterBuildProxy(OperatorHandler* ptrOpHandler, const WorkerThreadId workerThreadId)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    auto* opHandler = dynamic_cast<WindowBasedOperatorHandler*>(ptrOpHandler);
    if (!opHandler->enterBuild(workerThreadId))
    {
        /// Only reachable if a spill barrier stayed up past its timeout, which means the spilling side
        /// is wedged. Failing the task is the safe response: proceeding would write into a structure
        /// that may be mid-drain.
        throw CannotAllocateBuffer("WindowBuildPhysicalOperator: spill barrier did not lift; refusing to build into slice state");
    }
}

void exitBuildProxy(OperatorHandler* ptrOpHandler, const WorkerThreadId workerThreadId)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    dynamic_cast<WindowBasedOperatorHandler*>(ptrOpHandler)->exitBuild(workerThreadId);
}

void triggerAllWindowsProxy(OperatorHandler* ptrOpHandler, PipelineExecutionContext* piplineContext)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    PRECONDITION(piplineContext != nullptr, "pipeline context should not be null");

    auto* opHandler = dynamic_cast<WindowBasedOperatorHandler*>(ptrOpHandler);
    opHandler->triggerAllWindows(piplineContext);
}

/// The slice store needs to know in how many pipelines this operator appears, and consequently, how many terminations it will receive
void registerActivePipeline(OperatorHandler* ptrOpHandler, PipelineExecutionContext* pipelineCtx)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    PRECONDITION(pipelineCtx != nullptr, "pipeline context should not be null!");
    auto* opHandler = dynamic_cast<WindowBasedOperatorHandler*>(ptrOpHandler);
    /// The probe pipeline's setup wraps the store with the spilling decorator concurrently; synchronize
    /// via the handler's call_once before dereferencing it, so we never observe the moved-from store.
    opHandler->ensureSpillStoreInitialized(*pipelineCtx);
    opHandler->getSliceAndWindowStore().incrementNumberOfInputPipelines();
}

void WindowBuildPhysicalOperator::close(ExecutionContext& executionCtx, RecordBuffer&) const
{
    auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    /// Release the build bracket BEFORE triggering: checkAndTriggerWindows reaches into the slice
    /// store, and holding the bracket across it would keep a spill barrier waiting for a thread that
    /// is no longer writing anything.
    invoke(exitBuildProxy, operatorHandlerMemRef, executionCtx.workerThreadId);

    /// Update the watermark for the nlj operator and trigger slices
    invoke(
        checkWindowsTriggerProxy,
        operatorHandlerMemRef,
        executionCtx.pipelineContext,
        executionCtx.watermarkTs,
        executionCtx.sequenceNumber,
        executionCtx.chunkNumber,
        executionCtx.lastChunk,
        executionCtx.originId);
}

void WindowBuildPhysicalOperator::setup(ExecutionContext& executionCtx, CompilationContext&) const
{
    auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    invoke(registerActivePipeline, operatorHandler, executionCtx.pipelineContext);

    sliceStoreRef->setupSliceStore(executionCtx.pipelineContext);
}

nautilus::val<uint64_t> WindowBuildPhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    /// Initializing the time function
    timeFunction->open(executionCtx, recordBuffer);

    /// Creating the local state for the window operator build.
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    /// One atomic store per buffer when spill is on, one null check when it is off. Must bracket every
    /// path that writes slice state, so it goes here rather than around individual inserts.
    invoke(enterBuildProxy, operatorHandler, executionCtx.workerThreadId);
    executionCtx.setLocalOperatorState(id, std::make_unique<WindowOperatorBuildLocalState>(operatorHandler));
    return nautilus::val<uint64_t>{0};
}

void WindowBuildPhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    invoke(triggerAllWindowsProxy, operatorHandlerMemRef, executionCtx.pipelineContext);
}

std::optional<PhysicalOperator> WindowBuildPhysicalOperator::getChild() const
{
    return child;
}

void WindowBuildPhysicalOperator::setChild(PhysicalOperator child)
{
    this->child = std::move(child);
}

WindowBuildPhysicalOperator::WindowBuildPhysicalOperator(
    const OperatorHandlerId operatorHandlerId, std::unique_ptr<TimeFunction> timeFunction, std::unique_ptr<SliceStoreRef> sliceStoreRef)
    : operatorHandlerId(operatorHandlerId), timeFunction(std::move(timeFunction)), sliceStoreRef(std::move(sliceStoreRef))
{
}

WindowBuildPhysicalOperator::WindowBuildPhysicalOperator(const WindowBuildPhysicalOperator& other)
    : PhysicalOperatorConcept(other.id)
    , child(other.child)
    , operatorHandlerId(other.operatorHandlerId)
    , timeFunction(other.timeFunction ? other.timeFunction->clone() : nullptr)
    /// The SliceStoreRef is shared across pipeline copies — it manages per-pipeline caches internally
    , sliceStoreRef(other.sliceStoreRef->clone())
{
}
}
