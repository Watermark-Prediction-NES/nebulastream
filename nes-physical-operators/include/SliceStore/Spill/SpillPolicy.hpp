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

#include <cstdint>
#include <SliceStore/Slice.hpp>
#include <Time/Timestamp.hpp>

namespace NES
{

/// Read-only snapshot the policy uses to decide per-slice spill/restore/keep.
struct SliceSpillContext
{
    SliceEnd sliceEnd{Timestamp::INVALID_VALUE};
    Timestamp now{Timestamp::INVALID_VALUE};
    /// Wall-clock at which the predictor estimates the slice will be triggerable; INVALID_VALUE until predictor warm.
    Timestamp predictedTriggerWallClock{Timestamp::INVALID_VALUE};
    uint64_t residentBytes{0};
    uint64_t spilledBytes{0};
    WindowInfoState windowState{WindowInfoState::WINDOW_FILLING};
};

enum class SpillDecision : uint8_t
{
    Keep,
    Spill,
    Restore,
};

/// Policy interface for deciding whether to spill a slice to disk, restore it back, or leave it alone.
/// Called once per slice per GC tick; the global `memoryPressure` value [0.0, 1.0] is sampled once per tick.
class SpillPolicy
{
public:
    virtual ~SpillPolicy() = default;

    SpillPolicy() = default;
    SpillPolicy(const SpillPolicy&) = delete;
    SpillPolicy(SpillPolicy&&) = delete;
    SpillPolicy& operator=(const SpillPolicy&) = delete;
    SpillPolicy& operator=(SpillPolicy&&) = delete;

    /// Per-slice decision. Implementations are stateless wrt this call; observation state lives in observe().
    [[nodiscard]] virtual SpillDecision decide(const SliceSpillContext& ctx, double memoryPressure) const = 0;

    /// Called by the decorator at the start of every GC tick with the current global watermark.
    /// Default no-op; PressureSpillPolicy overrides to feed its WatermarkPredictor (when one is configured).
    virtual void observe(Timestamp /*now*/, Timestamp /*globalWatermark*/) noexcept { }
};

}
