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

/// Where a slice's state lives. Also the tag the slice store keeps per spilled slice so it knows which
/// storage backend to read back through. Ordered cheapest-to-restore first.
enum class SliceTier : uint8_t
{
    Resident,
    CompressedRam,
    CompressedDisk,
    Disk,
};

/// Read-only snapshot the policy uses to pick a slice's tier.
struct SliceSpillContext
{
    SliceEnd sliceEnd{Timestamp::INVALID_VALUE};
    /// Wall clock -- the SAME domain as WatermarkPredictor::predictWallClock, so a policy can subtract them.
    Timestamp now{Timestamp::INVALID_VALUE};
    SliceTier currentTier{SliceTier::Resident};
    uint64_t residentBytes{0};
    uint64_t spilledBytes{0};
    WindowInfoState windowState{WindowInfoState::WINDOW_FILLING};
};

/// Policy interface for placing a slice in a memory tier. Called once per slice per GC tick; the global
/// `memoryPressure` value [0.0, 1.0] is sampled once per tick. The policy returns the tier the slice
/// SHOULD be in; the store compares it against the current tier and moves the slice if they differ, so
/// keeping, demoting and promoting are all the same code path.
///
/// Returning `ctx.currentTier` means "leave it alone" -- note that is NOT the same as returning
/// `Resident`, which asks the store to restore an already-spilled slice.
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
    [[nodiscard]] virtual SliceTier decide(const SliceSpillContext& ctx, double memoryPressure) const = 0;

    /// Called by the decorator at the start of every GC tick with the current global watermark.
    /// Default no-op; PressureSpillPolicy overrides to feed its WatermarkPredictor (when one is configured).
    virtual void observe(Timestamp /*now*/, Timestamp /*globalWatermark*/) noexcept { }
};

}
