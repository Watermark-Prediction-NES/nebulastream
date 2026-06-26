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
#include <Configurations/SpillConfiguration.hpp>

namespace NES
{

/// Worker-level (CLI/YAML) defaults for the spill subsystem, exposed under
/// `worker.default_query_execution.spill.*`. Mirrors the SpillConfiguration POD field-for-field.
///
/// The POD SpillConfiguration stays the per-query lingua franca (it flows through the SQL binder and
/// logical plan as a plain copy). This BaseConfiguration is only the engine-default surface: the
/// QueryCompiler converts it via toSpillConfiguration() and a per-query `SET (SPILL.* AS ...)` clause
/// still fully overrides it.
class SpillWorkerConfiguration final : public BaseConfiguration
{
public:
    SpillWorkerConfiguration() = default;
    SpillWorkerConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { };

    BoolOption enabled = {"enabled", "false", "Enable the spill subsystem for all queries on this worker."};
    StringOption policyName = {"policy", "never", "Spill policy registered with SpillPolicyRegistry (e.g. REACTIVE, PREDICTIVE, NEVER)."};
    StringOption storageBackendName
        = {"backend", "in-memory", "Storage backend registered with StorageBackendRegistry (in-memory, local-file)."};
    StringOption spillDirectory = {"directory", "/tmp/nes-spill", "Directory for on-disk spill files (local-file backend)."};
    UIntOption storageIoThreads = {"io_threads", "4", "Thread count for the storage backend's executor."};
    FloatOption highMemoryBound = {"high_memory_bound", "0.85", "Spill above this memory-pressure ratio. 0.0 = always spill."};
    UIntOption predictionHorizonMs
        = {"prediction_horizon_ms", "50", "Predictive policy: ms ahead the predictor must say a slice triggers to keep it resident."};
    StringOption predictorName
        = {"predictor", "ewma", "Watermark predictor for the predictive policy (ewma, kalman, robustkalman)."};

    /// Materialise the POD consumed by the lowering rules / SliceStoreFactory.
    [[nodiscard]] SpillConfiguration toSpillConfiguration() const
    {
        return SpillConfiguration{
            .enabled = enabled.getValue(),
            .policyName = policyName.getValue(),
            .storageBackendName = storageBackendName.getValue(),
            .spillDirectory = spillDirectory.getValue(),
            .storageIoThreads = static_cast<uint32_t>(storageIoThreads.getValue()),
            .highMemoryBound = highMemoryBound.getValue(),
            .predictionHorizon = std::chrono::milliseconds{predictionHorizonMs.getValue()},
            .predictorName = predictorName.getValue(),
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
            &predictorName};
    }
};

}
