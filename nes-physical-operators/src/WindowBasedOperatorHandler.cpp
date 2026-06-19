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

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <Identifiers/Identifiers.hpp>
#include <Join/StreamJoinUtil.hpp>
#include <Runtime/QueryTerminationType.hpp>
#include <SliceStore/SliceStoreFactory.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Util/Logger/Logger.hpp>
#include <Watermark/MultiOriginWatermarkProcessor.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{

WindowBasedOperatorHandler::WindowBasedOperatorHandler(
    const std::vector<OriginId>& inputOrigins,
    const OriginId outputOriginId,
    std::unique_ptr<WindowSlicesStoreInterface> sliceAndWindowStore,
    SpillConfiguration spillConfig,
    std::string serializerName)
    : sliceAndWindowStore(std::move(sliceAndWindowStore))
    , watermarkProcessorBuild(std::make_unique<MultiOriginWatermarkProcessor>(inputOrigins))
    , watermarkProcessorProbe(std::make_unique<MultiOriginWatermarkProcessor>(std::vector{outputOriginId}))
    , numberOfWorkerThreads(0)
    , outputOriginId(outputOriginId)
    , inputOrigins(inputOrigins)
    , spillConfig(std::move(spillConfig))
    , serializerName(std::move(serializerName))
{
}

void WindowBasedOperatorHandler::start(PipelineExecutionContext& pipelineExecutionContext, uint32_t)
{
    numberOfWorkerThreads = pipelineExecutionContext.getNumberOfWorkerThreads();
    ensureSpillStoreInitialized(pipelineExecutionContext);
}

void WindowBasedOperatorHandler::ensureSpillStoreInitialized(PipelineExecutionContext& pipelineExecutionContext)
{
    if (!spillConfig.enabled)
    {
        return;
    }
    /// Deferred wrap: at lowering time the buffer provider was not available, so the lowering rule
    /// constructed a plain DefaultTimeBasedSliceStore. Now we have a real buffer manager via the
    /// pipeline context; wrap the in-memory store with the spilling decorator. Build- and probe-pipeline
    /// setups run concurrently and both reach the store, so the wrap must happen exactly once and
    /// before any concurrent reader observes the moved-from `sliceAndWindowStore`. call_once gives both:
    /// the first caller wraps while the others block until it completes.
    std::call_once(
        spillWrapOnceFlag,
        [&]
        {
            const auto bufferProvider = pipelineExecutionContext.getBufferManager();
            if (bufferProvider != nullptr)
            {
                sliceAndWindowStore
                    = SliceStoreFactory::wrapWithSpill(std::move(sliceAndWindowStore), spillConfig, bufferProvider.get(), serializerName);
            }
            else
            {
                NES_WARNING("WindowBasedOperatorHandler: spill enabled but no buffer manager available; staying in memory");
            }
        });
}

void WindowBasedOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext&)
{
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

    /// Getting all slices that can be triggered and triggering them
    const auto slicesAndWindowInfo = sliceAndWindowStore->getTriggerableWindowSlices(newGlobalWatermark);
    triggerSlices(slicesAndWindowInfo, pipelineCtx);
}

void WindowBasedOperatorHandler::triggerAllWindows(PipelineExecutionContext* pipelineCtx)
{
    const auto slicesAndWindowInfo = sliceAndWindowStore->getAllNonTriggeredSlices();
    NES_TRACE("Triggering {} windows for origin: {}", slicesAndWindowInfo.size(), outputOriginId);
    triggerSlices(slicesAndWindowInfo, pipelineCtx);
}

}
