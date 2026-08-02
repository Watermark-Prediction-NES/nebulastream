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

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/ScalarOption.hpp>
#include <Configurations/SliceEvictionConfiguration.hpp>

namespace NES
{

/// Worker-level (CLI/YAML) defaults for slice eviction — spilling and compression alike — exposed under
/// `worker.default_query_execution.eviction.*`. Mirrors the SliceEvictionConfiguration POD field-for-field.
///
/// The mirror is unavoidable: BaseConfiguration needs one BaseOption member per CLI/YAML key, and the
/// POD stays a plain struct because it travels per query through the SQL binder and LogicalPlan. It is
/// also not a pure copy — several fields change representation across the boundary (`_ms` UIntOptions
/// become std::chrono::milliseconds, `policy`/`backend`/`predictor` drop the `Name` suffix), which is
/// why toSliceEvictionConfiguration() below is the single place that knows the mapping.
///
/// This is the surface the systest eviction matrix drives (nes-systests/systest/CMakeLists.txt): the
/// `.test` files carry no SET (EVICTION.*) clause, so the same queries run under every policy variant.
///
/// CAVEAT: a per-query `SET (EVICTION.* AS ...)` clause replaces the resulting POD WHOLESALE, so naming
/// any single EVICTION.* key discards every option below and falls back to SliceEvictionConfiguration's
/// struct defaults. These options therefore only apply to queries with no SET eviction clause at all.
/// See the KNOWN LIMITATION comment in QueryCompiler::compileQuery.
class SliceEvictionWorkerConfiguration final : public BaseConfiguration
{
public:
    SliceEvictionWorkerConfiguration() = default;
    SliceEvictionWorkerConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { };

    BoolOption enabled = {"enabled", "false", "Enable slice eviction (spill and/or compression) for all queries on this worker."};
    StringOption policyName
        = {"policy", "never", "Spill policy registered with SpillPolicyRegistry (NEVER, REACTIVE, ALWAYS, PREDICTIVE, TIERED)."};
    StringOption storageBackendName
        = {"backend", "in-memory", "Storage backend registered with StorageBackendRegistry (in-memory, local-file)."};
    StringOption spillDirectory = {"directory", "/tmp/nes-spill", "Directory for on-disk spill files (local-file backend)."};
    UIntOption storageIoThreads = {"io_threads", "4", "Thread count for the storage backend's executor."};
    FloatOption highMemoryBound = {"high_memory_bound", "0.85", "Spill above this memory-pressure ratio. 0.0 = always spill."};
    UIntOption predictionHorizonMs
        = {"prediction_horizon_ms", "50", "Predictive policy: ms ahead the predictor must say a slice triggers to keep it resident."};
    StringOption predictorName = {"predictor", "ewma", "Watermark predictor for the predictive policy (ewma, kalman, robustkalman)."};
    UIntOption promoteHorizonMs
        = {"promote_horizon_ms", "20", "Tiered policy: restore a spilled slice eagerly once its predicted trigger is this close."};
    UIntOption compressRamHorizonMs
        = {"compress_ram_horizon_ms", "200", "Tiered policy: compress in RAM while the predicted trigger is within this many ms."};
    UIntOption compressDiskHorizonMs
        = {"compress_disk_horizon_ms", "1000", "Tiered policy: compress to disk within this many ms; beyond it, spill raw."};
    BoolOption compress = {"compress", "false", "Compress evicted slice bytes (zstd) before they reach the storage backend."};
    UIntOption compressionLevel = {"compression_level", "3", "zstd compression level (1-22), used when compress is true."};

    /// Materialise the POD consumed by the lowering rules / SliceStoreFactory.
    [[nodiscard]] SliceEvictionConfiguration toSliceEvictionConfiguration() const
    {
        return SliceEvictionConfiguration{
            .enabled = enabled.getValue(),
            .policyName = policyName.getValue(),
            .storageBackendName = storageBackendName.getValue(),
            .spillDirectory = spillDirectory.getValue(),
            .storageIoThreads = static_cast<uint32_t>(storageIoThreads.getValue()),
            .highMemoryBound = highMemoryBound.getValue(),
            .predictionHorizon = std::chrono::milliseconds{predictionHorizonMs.getValue()},
            /// Designator order must match SliceEvictionConfiguration's declaration order (-Werror=reorder-init-list).
            .promoteHorizon = std::chrono::milliseconds{promoteHorizonMs.getValue()},
            .compressRamHorizon = std::chrono::milliseconds{compressRamHorizonMs.getValue()},
            .compressDiskHorizon = std::chrono::milliseconds{compressDiskHorizonMs.getValue()},
            .predictorName = predictorName.getValue(),
            .compress = compress.getValue(),
            .compressionLevel = static_cast<uint32_t>(compressionLevel.getValue()),
        };
    }

private:
    std::vector<BaseOption*> getOptions() override
    {
        return {
            &enabled,
            &policyName,
            &storageBackendName,
            &spillDirectory,
            &storageIoThreads,
            &highMemoryBound,
            &predictionHorizonMs,
            &predictorName,
            &promoteHorizonMs,
            &compressRamHorizonMs,
            &compressDiskHorizonMs,
            &compress,
            &compressionLevel};
    }
};

}
