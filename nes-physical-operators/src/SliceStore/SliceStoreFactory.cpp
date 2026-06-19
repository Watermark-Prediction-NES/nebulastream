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

#include <SliceStore/SliceStoreFactory.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/DefaultTimeBasedSliceStore.hpp>
#include <SliceStore/Spill/BufferPoolPressureSensor.hpp>
#include <SliceStore/Spill/MemoryPressureSensor.hpp>
#include <SliceStore/Spill/SpillConfiguration.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/Spill/StorageBackendRegistry.hpp>
#include <SliceStore/SpillingTimeBasedSliceStore.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <SliceCacheConfiguration.hpp>
#include <SpillPolicyRegistry.hpp>

namespace NES
{

std::unique_ptr<WindowSlicesStoreInterface> SliceStoreFactory::create(
    const SpillConfiguration& spillConfig,
    uint64_t windowSize,
    uint64_t windowSlide,
    SliceCacheConfiguration sliceCacheConfig,
    AbstractBufferProvider* bufferProvider)
{
    auto inner = std::make_unique<DefaultTimeBasedSliceStore>(windowSize, windowSlide, sliceCacheConfig);
    if (!spillConfig.enabled)
    {
        return inner;
    }

    PRECONDITION(bufferProvider != nullptr, "Spill enabled but no buffer provider supplied to SliceStoreFactory");

    auto policy = SpillPolicyRegistry::instance()
                      .create(
                          spillConfig.policyName,
                          SpillPolicyRegistryArguments{
                              .lowMemoryBound = spillConfig.lowMemoryBound,
                              .highMemoryBound = spillConfig.highMemoryBound,
                              .horizon = spillConfig.predictionHorizon,
                              .predictor = nullptr,
                          })
                      .value_or(nullptr);
    if (!policy)
    {
        NES_WARNING("SliceStoreFactory: unknown spill policy '{}', falling back to never-spill", spillConfig.policyName);
        policy = SpillPolicyRegistry::instance().create("never", SpillPolicyRegistryArguments{}).value_or(nullptr);
    }

    auto backend = StorageBackendRegistry::instance().create(
        spillConfig.storageBackendName,
        StorageBackendArgs{.spillDirectory = spillConfig.spillDirectory, .ioThreads = spillConfig.storageIoThreads});
    if (!backend)
    {
        NES_WARNING("SliceStoreFactory: unknown storage backend '{}', falling back to in-memory", spillConfig.storageBackendName);
        backend = StorageBackendRegistry::instance().create("in-memory", StorageBackendArgs{});
    }
    PRECONDITION(backend != nullptr, "No storage backend available even after fallback to in-memory");

    auto sensor = std::make_unique<BufferPoolPressureSensor>(*bufferProvider, /*capacity*/ bufferProvider->getBufferSize());

    return std::make_unique<SpillingTimeBasedSliceStore>(
        std::move(inner), std::move(policy), std::move(backend), std::move(sensor), *bufferProvider);
}

}
