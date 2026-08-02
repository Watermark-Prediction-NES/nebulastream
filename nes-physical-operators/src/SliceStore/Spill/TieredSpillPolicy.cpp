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

#include <SliceStore/Spill/TieredSpillPolicy.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string_view>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Time/Timestamp.hpp>
#include <Util/Logger/Logger.hpp>
#include <Watermark/WatermarkPredictorFactory.hpp>
#include <SpillPolicyRegistry.hpp>

namespace NES
{

TieredSpillPolicy::TieredSpillPolicy(
    const double highBound_,
    const std::chrono::milliseconds promoteHorizon_,
    const std::chrono::milliseconds nearHorizon_,
    const std::chrono::milliseconds compressRamHorizon_,
    const std::chrono::milliseconds compressDiskHorizon_,
    const std::string_view predictorName) noexcept
    : highBound(highBound_)
    , promoteHorizon(promoteHorizon_)
    , nearHorizon(std::max(nearHorizon_, promoteHorizon_))
    , compressRamHorizon(std::max(compressRamHorizon_, std::max(nearHorizon_, promoteHorizon_)))
    , compressDiskHorizon(std::max(compressDiskHorizon_, std::max(compressRamHorizon_, std::max(nearHorizon_, promoteHorizon_))))
    , predictor(makeWatermarkPredictor(predictorName))
{
    /// Out-of-order horizons would silently make a band unreachable. Clamping keeps decide() total and
    /// deterministic; warn so a bad config still surfaces.
    if (nearHorizon != nearHorizon_ || compressRamHorizon != compressRamHorizon_ || compressDiskHorizon != compressDiskHorizon_)
    {
        NES_WARNING(
            "TieredSpillPolicy: horizons must be non-decreasing (promote <= near <= compressRam <= compressDisk); "
            "clamped to {}/{}/{}/{} ms",
            promoteHorizon.count(),
            nearHorizon.count(),
            compressRamHorizon.count(),
            compressDiskHorizon.count());
    }
}

SliceTier TieredSpillPolicy::decide(const SliceSpillContext& ctx, const double memoryPressure) const
{
    /// Time until the predictor thinks the probe arrives. Both operands are wall clock.
    /// INVALID_VALUE == the predictor cannot answer yet.
    uint64_t timeToTrigger = Timestamp::INVALID_VALUE;
    if (predictor)
    {
        if (const auto predicted = predictor->predictWallClock(ctx.sliceEnd); predicted.getRawValue() != Timestamp::INVALID_VALUE)
        {
            const auto nowRaw = ctx.now.getRawValue();
            const auto predRaw = predicted.getRawValue();
            timeToTrigger = predRaw <= nowRaw ? 0 : predRaw - nowRaw;
        }
    }
    const bool haveEstimate = timeToTrigger != Timestamp::INVALID_VALUE;

    /// Promotion is deliberately evaluated BEFORE the pressure gate: a slice about to be probed must be
    /// resident whether or not memory is tight, otherwise the probe pays restore latency. This is also
    /// what makes the ladder non-oscillating — time-to-trigger only shrinks as the clock advances, so a
    /// slice that enters the promote band never leaves it.
    if (haveEstimate && timeToTrigger <= static_cast<uint64_t>(promoteHorizon.count()))
    {
        return SliceTier::Resident;
    }

    /// Below the gate nothing moves. Same contract as PressureSpillPolicy: memory is fine, so leave the
    /// slice wherever it is and let the probe pull it back.
    if (memoryPressure < highBound)
    {
        return ctx.currentTier;
    }

    /// Cold predictor: byte-identical to the reactive policy.
    if (!haveEstimate)
    {
        return SliceTier::Disk;
    }

    if (timeToTrigger <= static_cast<uint64_t>(nearHorizon.count()))
    {
        return SliceTier::Resident;
    }
    if (timeToTrigger <= static_cast<uint64_t>(compressRamHorizon.count()))
    {
        return SliceTier::CompressedRam;
    }
    if (timeToTrigger <= static_cast<uint64_t>(compressDiskHorizon.count()))
    {
        return SliceTier::CompressedDisk;
    }
    return SliceTier::Disk;
}

/// Same shape as PressureSpillPolicy::observe — feed the predictor once per distinct watermark. Kept as
/// a copy rather than hoisted into SpillPolicy: ten lines do not justify a shared base with state.
void TieredSpillPolicy::observe(const Timestamp now, const Timestamp globalWatermark) noexcept
{
    if (!predictor || globalWatermark.getRawValue() == Timestamp::INVALID_VALUE)
    {
        return;
    }
    if (lastObservedWatermark.getRawValue() != Timestamp::INVALID_VALUE && globalWatermark == lastObservedWatermark)
    {
        return;
    }
    predictor->observe(globalWatermark, now);
    lastObservedWatermark = globalWatermark;
}

SpillPolicyRegistryReturnType SpillPolicyGeneratedRegistrar::RegisterTIEREDSpillPolicy(SpillPolicyRegistryArguments args)
{
    return std::make_unique<TieredSpillPolicy>(
        args.highMemoryBound, args.promoteHorizon, args.horizon, args.compressRamHorizon, args.compressDiskHorizon, args.predictorName);
}

}
