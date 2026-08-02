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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <Configurations/SliceEvictionConfiguration.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Spill/BufferPoolPressureSensor.hpp>
#include <SliceStore/Spill/CompressionStorageBackend.hpp>
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

namespace
{
/// Registry names are case-insensitive, so match the same way the registry does.
bool policyIsTiered(std::string_view policyName)
{
    return std::ranges::equal(policyName, std::string_view{"tiered"}, [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}
}

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
    (void)forceLinkCompressionStorageBackend();

    if (!evictionConfig.enabled)
    {
        return inner;
    }
    if (bufferProvider == nullptr)
    {
        NES_WARNING("SliceStoreFactory::wrapWithEviction: eviction enabled but buffer provider is null; keeping in-memory store");
        return inner;
    }

    const SpillPolicyRegistryArguments policyArgs{
        .highMemoryBound = evictionConfig.highMemoryBound,
        .horizon = evictionConfig.predictionHorizon,
        .promoteHorizon = evictionConfig.promoteHorizon,
        .compressRamHorizon = evictionConfig.compressRamHorizon,
        .compressDiskHorizon = evictionConfig.compressDiskHorizon,
        .predictor = nullptr,
        .predictorName = evictionConfig.predictorName,
    };
    auto policy = SpillPolicyRegistry::instance().create(evictionConfig.policyName, policyArgs).value_or(nullptr);
    if (!policy)
    {
        throw UnknownSpillPolicy("SliceStoreFactory::wrapWithEviction: unknown spill policy '{}'", evictionConfig.policyName);
    }

    /// `compress` and `storageBackendName` are orthogonal knobs under every policy.
    ///
    /// Non-tiered policies use the Disk tier alone, so the two axes give four modes directly:
    /// (in-memory, off/on) and (local-file, off/on) -- in particular in-memory + compress is
    /// "compress without spilling", and local-file + no compress is "spill without compressing".
    ///
    /// The tiered policy expresses compression as a TIER instead, so it keeps its Disk tier raw and
    /// lets `compress` decide whether the compressed rungs exist at all (below). Letting the Disk tier
    /// compress here as well would make it identical to CompressedDisk apart from its directory, i.e.
    /// a four-rung ladder with only three distinct behaviours.
    const bool tiered = policyIsTiered(evictionConfig.policyName);
    const StorageBackendArgs backendArgs{
        .spillDirectory = evictionConfig.spillDirectory,
        .ioThreads = evictionConfig.storageIoThreads,
        .innerBackendName = evictionConfig.storageBackendName,
        .compressionLevel = evictionConfig.compressionLevel};
    const auto backendName = (evictionConfig.compress && !tiered) ? std::string{"compression"} : evictionConfig.storageBackendName;
    auto backend = StorageBackendRegistry::instance().create(backendName, backendArgs);
    if (!backend)
    {
        throw UnknownStorageBackend("SliceStoreFactory::wrapWithEviction: unknown storage backend '{}'", backendName);
    }

    SpillingTimeBasedSliceStore::TierBackends tierBackends{};
    tierBackends[static_cast<std::size_t>(SliceTier::Disk)] = std::move(backend);

    /// Only the tiered policy can ask for the compressed tiers, and only when compression is enabled:
    /// with `compress` off the ladder degrades to Resident <-> Disk, so prediction still decides WHEN a
    /// slice is evicted but never compresses it. TieredSpillPolicy may still name a compressed tier;
    /// the store leaves the slice where it is when that tier has no backend. Gating here also matters
    /// because LocalFileStorageBackend's constructor creates its directory unconditionally, so building
    /// these eagerly would litter the spill directory for every query that never uses them.
    if (tiered && evictionConfig.compress)
    {
        /// CompressedRam is always RAM-resident: that IS the tier. CompressedDisk follows the configured
        /// backend, exactly as the Disk tier above does, so `backend=in-memory` keeps the whole ladder off
        /// disk (no files to collide over when queries share a spill directory) while `backend=local-file`
        /// gives the real three-level hierarchy.
        tierBackends[static_cast<std::size_t>(SliceTier::CompressedRam)] = StorageBackendRegistry::instance().create(
            "compression",
            StorageBackendArgs{
                .spillDirectory = evictionConfig.spillDirectory,
                .ioThreads = evictionConfig.storageIoThreads,
                .innerBackendName = "in-memory",
                .compressionLevel = evictionConfig.compressionLevel});
        /// LocalFileStorageBackend derives its filenames from the SpillObjectKey alone, which carries no
        /// tier. Two local-file-backed tiers sharing a directory would therefore map a slice to the SAME
        /// file, and a CompressedDisk -> Disk move would read and write it at once. Separate directories.
        tierBackends[static_cast<std::size_t>(SliceTier::CompressedDisk)] = StorageBackendRegistry::instance().create(
            "compression",
            StorageBackendArgs{
                .spillDirectory = std::filesystem::path{evictionConfig.spillDirectory} / "compressed",
                .ioThreads = evictionConfig.storageIoThreads,
                .innerBackendName = evictionConfig.storageBackendName,
                .compressionLevel = evictionConfig.compressionLevel});
    }

    /// Resolve the serializer once, here, from the name the handler was given in the lowering.
    /// A missing entry means this Slice subclass does not support spilling — keep the in-memory store.
    auto* serializer = SliceStateSerializerRegistry::instance().lookup(serializerName);
    if (serializer == nullptr)
    {
        NES_WARNING("SliceStoreFactory::wrapWithEviction: no serializer registered for '{}'; keeping in-memory store", serializerName);
        return inner;
    }

    auto sensor = std::make_unique<BufferPoolPressureSensor>(*bufferProvider);

    return std::make_unique<SpillingTimeBasedSliceStore>(
        std::move(inner), std::move(policy), std::move(tierBackends), std::move(sensor), *bufferProvider, *serializer);
}

}
