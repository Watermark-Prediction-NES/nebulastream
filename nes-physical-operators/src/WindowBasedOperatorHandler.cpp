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

#include <WindowBasedOperatorHandler.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/QueryTerminationType.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <StateReduction/StateReductionManager.hpp>
#include <Util/Logger/Logger.hpp>
#include <Watermark/MultiOriginWatermarkProcessor.hpp>
#include <PipelineExecutionContext.hpp>
#include <StateReductionConfiguration.hpp>

namespace NES
{

namespace
{
/// Distinguishes the spill directories of two operators in the same process. OperatorHandlerId is not
/// usable here: handlers do not carry their own id.
uint64_t nextStateReductionInstanceId()
{
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}
}

WindowBasedOperatorHandler::WindowBasedOperatorHandler(
    const std::vector<OriginId>& inputOrigins,
    const OriginId outputOriginId,
    std::unique_ptr<WindowSlicesStoreInterface> sliceAndWindowStore,
    const StateReductionConfiguration& stateReductionConfiguration)
    : sliceAndWindowStore(std::move(sliceAndWindowStore))
    /// this->sliceAndWindowStore is declared before stateReduction, so it is already initialised here.
    , stateReduction(std::make_unique<StateReductionManager>(
          stateReductionConfiguration, nextStateReductionInstanceId(), this->sliceAndWindowStore->getWindowSize()))
    , watermarkProcessorBuild(std::make_unique<MultiOriginWatermarkProcessor>(inputOrigins))
    , watermarkProcessorProbe(std::make_unique<MultiOriginWatermarkProcessor>(std::vector{outputOriginId}))
    , outputOriginId(outputOriginId)
    , inputOrigins(inputOrigins)
{
}

void WindowBasedOperatorHandler::start(PipelineExecutionContext& pipelineExecutionContext, uint32_t)
{
    numberOfWorkerThreads = pipelineExecutionContext.getNumberOfWorkerThreads();
    /// Measures what compression and spilling cost on this machine. Until this runs, a cost-model
    /// predictor deliberately reduces nothing, so it has to happen before any state is built up.
    stateReduction->calibrate();
}

void WindowBasedOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext&)
{
    /// Last point at which this operator's numbers still exist. Writes one CSV row if a stats path was
    /// configured, and does nothing otherwise; calling it twice writes once.
    stateReduction->writeStats();

    /// The manager itself is deliberately left alive until the handler is destroyed. It is what owns the
    /// spill directory, and tearing it down here would only mean null-checking it on every other path.
}

void WindowBasedOperatorHandler::pinSlice(const std::shared_ptr<Slice>& slice, AbstractBufferProvider& bufferProvider) const
{
    stateReduction->pin(slice, bufferProvider);
}

void WindowBasedOperatorHandler::pinSlices(const std::vector<std::shared_ptr<Slice>>& slices, AbstractBufferProvider& bufferProvider) const
{
    for (const auto& slice : slices)
    {
        if (slice)
        {
            pinSlice(slice, bufferProvider);
        }
    }
}

WindowSlicesStoreInterface& WindowBasedOperatorHandler::getSliceAndWindowStore() const
{
    return *sliceAndWindowStore;
}

void WindowBasedOperatorHandler::garbageCollectSlicesAndWindows(const BufferMetaData& bufferMetaData) const
{
    const auto newGlobalWaterMarkProbe
        = watermarkProcessorProbe->updateWatermark(bufferMetaData.watermarkTs, bufferMetaData.seqNumber, bufferMetaData.originId);

    NES_TRACE(
        "New global watermark probe: {} for origin: {} and sequence data: {} and watermarkTs of buffer {}",
        newGlobalWaterMarkProbe,
        bufferMetaData.originId,
        bufferMetaData.seqNumber,
        bufferMetaData.watermarkTs);
    sliceAndWindowStore->garbageCollectSlicesAndWindows(newGlobalWaterMarkProbe);
}

void WindowBasedOperatorHandler::checkAndTriggerWindows(const BufferMetaData& bufferMetaData, PipelineExecutionContext* pipelineCtx)
{
    /// The watermark processor handles the minimal watermark across both streams
    const auto newGlobalWatermark
        = watermarkProcessorBuild->updateWatermark(bufferMetaData.watermarkTs, bufferMetaData.seqNumber, bufferMetaData.originId);

    NES_TRACE(
        "New global watermark: {} for origin: {} and sequence data: {} and watermarkTs of buffer {}",
        newGlobalWatermark,
        bufferMetaData.originId,
        bufferMetaData.seqNumber,
        bufferMetaData.watermarkTs);

    /// The one place every window operator sees a fresh global build watermark, and therefore the only
    /// place that can decide what to do with slice state. Driven off actual advances rather than off
    /// every input buffer, because enumerating the store's slices costs a lock and a copy.
    if (stateReduction->isEnabled()
        and lastReductionWatermark.exchange(newGlobalWatermark.getRawValue(), std::memory_order_relaxed)
            != newGlobalWatermark.getRawValue())
    {
        stateReduction->onWatermarkAdvanced(newGlobalWatermark, sliceAndWindowStore->getAllSlices());
    }

    /// Getting all slices that can be triggered and triggering them
    const auto slicesAndWindowInfo = sliceAndWindowStore->getTriggerableWindowSlices(newGlobalWatermark);
    triggerSlices(slicesAndWindowInfo, pipelineCtx);
}

void WindowBasedOperatorHandler::triggerAllWindows(PipelineExecutionContext* pipelineCtx)
{
    const auto slicesAndWindowInfo = sliceAndWindowStore->getAllNonTriggeredSlices();
    NES_TRACE("Triggering {} windows for origin: {}", slicesAndWindowInfo.size(), outputOriginId);

    /// Termination. Every slice that has not been triggered yet is about to be read one last time, so
    /// anything parked has to come back first — otherwise a query that ends with reduced slices silently
    /// drops their results. Nothing reduces after this point, so the pins simply stay.
    if (stateReduction->isEnabled())
    {
        auto bufferProvider = pipelineCtx->getBufferManager();
        for (const auto& windowSlices : slicesAndWindowInfo | std::views::values)
        {
            pinSlices(windowSlices, *bufferProvider);
        }
    }

    triggerSlices(slicesAndWindowInfo, pipelineCtx);
}

}
