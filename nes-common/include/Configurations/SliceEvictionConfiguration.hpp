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
/// POD configuration for slice eviction: moving a slice off the resident tier, whether by compressing it
/// in RAM, writing it to disk, or both. Engine-wide (mirrors SliceCacheConfiguration).
/// A plain struct rather than a BaseConfiguration because this value travels per query through the SQL
/// binder and LogicalPlan (which is copied on every optimizer rewrite): it needs cheap copies, designated
/// init, and no polymorphic base. SliceEvictionWorkerConfiguration mirrors these fields as CLI/YAML
/// options and converts them here — see the comment there for why that mirror cannot be collapsed.
/// `enabled == false` short-circuits the SliceStoreFactory back to the plain DefaultTimeBasedSliceStore,
/// so eviction has zero runtime cost when unused.
struct SliceEvictionConfiguration
{
    bool enabled{false};

    /// Name registered with SpillPolicyRegistry. Recognised values: "never" (no-op, the default),
    /// "reactive" (evict above highMemoryBound), "always" (evict every tick, bound ignored),
    /// "predictive" (reactive, but keep a slice resident when the predictor says its trigger is near),
    /// "tiered" (pick a tier per slice from the predicted time-to-trigger).
    std::string policyName{"never"};

    /// Name registered with StorageBackendRegistry.
    std::string storageBackendName{"in-memory"};

    /// Directory for on-disk spill files (used by local-file backend).
    std::string spillDirectory{"/tmp/nes-spill"};

    /// Thread count for the storage backend's executor.
    uint32_t storageIoThreads{4};

    /// Pressure threshold: spill above high.
    double highMemoryBound{0.85};

    /// Predictive-policy horizon — how many ms ahead the predictor must say the slice will trigger
    /// before the policy decides to keep it resident. Doubles as the tiered policy's "near" band.
    std::chrono::milliseconds predictionHorizon{50};

    /// Tiered-policy bands, in ms of predicted time-to-trigger. Must be non-decreasing with
    /// `predictionHorizon`; TieredSpillPolicy clamps and warns if they are not. Ignored by other policies.
    /// Inside `promoteHorizon` a spilled slice is restored eagerly, before the probe asks for it.
    std::chrono::milliseconds promoteHorizon{20};
    std::chrono::milliseconds compressRamHorizon{200};
    std::chrono::milliseconds compressDiskHorizon{1000};

    /// Watermark predictor selection for `policyName == "predictive"`. Recognised values: "ewma",
    /// "kalman", "robustkalman". Other policies ignore this field.
    std::string predictorName{"ewma"};

    /// Compress evicted slice bytes (zstd). Orthogonal to `storageBackendName` under every policy, but
    /// the two express it differently:
    ///   - never/reactive/predictive: wraps the configured backend, giving four modes --
    ///     in-memory + compress == compress without spilling; local-file without compress == spill
    ///     without compressing.
    ///   - tiered: brings the CompressedRam / CompressedDisk rungs into existence. With `compress`
    ///     false the ladder degrades to Resident <-> Disk, so prediction still decides WHEN a slice is
    ///     evicted but nothing is compressed. The tiered Disk rung is always raw, so it never
    ///     duplicates CompressedDisk.
    bool compress{false};

    /// zstd compression level (1–22) used when `compress` is true.
    uint32_t compressionLevel{3};
};

}
