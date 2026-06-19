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

namespace NES
{

/// POD configuration for the spill subsystem. Engine-wide (mirrors SliceCacheConfiguration).
/// `enabled == false` short-circuits the SliceStoreFactory back to the plain DefaultTimeBasedSliceStore,
/// so the spill subsystem has zero runtime cost when unused.
struct SpillConfiguration
{
    bool enabled{false};

    /// Name registered with SpillPolicyRegistry. Default: NeverSpillPolicy (no-op).
    std::string policyName{"never"};

    /// Name registered with StorageBackendRegistry.
    std::string storageBackendName{"in-memory"};

    /// Directory for on-disk spill files (used by local-file backend).
    std::string spillDirectory{"/tmp/nes-spill"};

    /// Thread count for the storage backend's executor.
    uint32_t storageIoThreads{4};

    /// Reactive policy thresholds. Spill above high; restore (if applicable) below low.
    double lowMemoryBound{0.6};
    double highMemoryBound{0.85};

    /// Predictive-policy horizon — how many ms ahead the predictor must say the slice will trigger
    /// before the policy decides to keep it resident.
    std::chrono::milliseconds predictionHorizon{50};
};

}
