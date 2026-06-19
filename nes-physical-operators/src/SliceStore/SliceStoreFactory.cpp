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

#include <memory>
#include <string>
#include <utility>
#include <Configurations/SliceEvictionConfiguration.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Spill/BufferPoolPressureSensor.hpp>
#include <SliceStore/Spill/HJSliceStateSerializer.hpp>
#include <SliceStore/Spill/InMemoryStorageBackend.hpp>
#include <SliceStore/Spill/LocalFileStorageBackend.hpp>
#include <SliceStore/Spill/MemoryPressureSensor.hpp>
#include <SliceStore/Spill/NLJSliceStateSerializer.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/SliceStateSerializerRegistry.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/Spill/StorageBackendRegistry.hpp>
#include <SliceStore/SpillingTimeBasedSliceStore.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <SpillPolicyRegistry.hpp>

namespace NES
{

std::unique_ptr<WindowSlicesStoreInterface> SliceStoreFactory::wrapWithEviction(
    std::unique_ptr<WindowSlicesStoreInterface> inner,
    const SliceEvictionConfiguration& evictionConfig,
    AbstractBufferProvider* bufferProvider,
    const std::string& serializerName)
{
    /// Ensure the per-slice serializer AND storage-backend registrars (all anonymous-namespace
    /// globals) are not dropped by the static-library linker when no other call site references them.
    /// Without this, e.g. the systest binary links neither backend and `create("in-memory")` returns
    /// null at runtime. Cheap calls.
    (void)forceLinkNLJSerializer();
    (void)forceLinkHJSerializer();
    (void)forceLinkInMemoryStorageBackend();
    (void)forceLinkLocalFileStorageBackend();

    if (!evictionConfig.enabled)
    {
        return inner;
    }
    if (bufferProvider == nullptr)
    {
        NES_WARNING("SliceStoreFactory::wrapWithEviction: spill enabled but buffer provider is null; keeping in-memory store");
        return inner;
    }

    const SpillPolicyRegistryArguments policyArgs{
        .highMemoryBound = evictionConfig.highMemoryBound,
        .horizon = evictionConfig.predictionHorizon,
        .predictor = nullptr,
        .predictorName = evictionConfig.predictorName,
    };
    auto policy = SpillPolicyRegistry::instance().create(evictionConfig.policyName, policyArgs).value_or(nullptr);
    if (!policy)
    {
        throw UnknownSpillPolicy("SliceStoreFactory::wrapWithEviction: unknown spill policy '{}'", evictionConfig.policyName);
    }

    auto backend = StorageBackendRegistry::instance().create(
        evictionConfig.storageBackendName,
        StorageBackendArgs{.spillDirectory = evictionConfig.spillDirectory, .ioThreads = evictionConfig.storageIoThreads});
    if (!backend)
    {
        throw UnknownStorageBackend("SliceStoreFactory::wrapWithEviction: unknown storage backend '{}'", evictionConfig.storageBackendName);
    }

    /// Resolve the serializer once, here, from the name the handler was given in the lowering.
    /// A missing entry means this Slice subclass does not support spilling — keep the in-memory store.
    auto* serializer = SliceStateSerializerRegistry::instance().lookup(serializerName);
    if (serializer == nullptr)
    {
        NES_WARNING("SliceStoreFactory::wrapWithEviction: no serializer registered for '{}'; keeping in-memory store", serializerName);
        return inner;
    }

    auto sensor = std::make_unique<BufferPoolPressureSensor>(*bufferProvider, /*capacity*/ bufferProvider->getBufferSize());

    return std::make_unique<SpillingTimeBasedSliceStore>(
        std::move(inner), std::move(policy), std::move(backend), std::move(sensor), *bufferProvider, *serializer);
}

}
