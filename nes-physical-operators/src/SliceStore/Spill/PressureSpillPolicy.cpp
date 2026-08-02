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

#include <SliceStore/Spill/PressureSpillPolicy.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string_view>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Time/Timestamp.hpp>
#include <Watermark/WatermarkPredictorFactory.hpp>
#include <SpillPolicyRegistry.hpp>

namespace NES
{

PressureSpillPolicy::PressureSpillPolicy(double highBound_) noexcept : highBound(highBound_)
{
}

PressureSpillPolicy::PressureSpillPolicy(double highBound_, std::chrono::milliseconds horizon_, std::string_view predictorName) noexcept
    : highBound(highBound_), horizon(horizon_), predictor(makeWatermarkPredictor(predictorName))
{
}

SliceTier PressureSpillPolicy::decide(const SliceSpillContext& ctx, double memoryPressure) const
{
    /// This policy only ever moves a slice one way, Resident -> Disk. Its "keep" answer is therefore
    /// ctx.currentTier, not Resident: an already-spilled slice must stay spilled until the probe pulls
    /// it back, which is the behaviour this policy has always had.
    ///
    /// Pressure guard: under-pressure cases short-circuit; the predictor is consulted only when memory is constrained.
    if (memoryPressure < highBound)
    {
        return ctx.currentTier;
    }

    /// No predictor, or one too cold to answer: pure-pressure behaviour — just spill.
    if (!predictor)
    {
        return SliceTier::Disk;
    }
    const Timestamp predictedTrigger = predictor->predictWallClock(ctx.sliceEnd);
    if (predictedTrigger.getRawValue() == Timestamp::INVALID_VALUE)
    {
        return SliceTier::Disk;
    }

    /// If the predicted trigger is further than `horizon` from now, spill. Otherwise keep the slice
    /// resident so that the probe doesn't pay restore latency on an imminent trigger.
    const auto nowRaw = ctx.now.getRawValue();
    const auto predRaw = predictedTrigger.getRawValue();
    if (predRaw <= nowRaw)
    {
        return ctx.currentTier;
    }
    const auto delta = predRaw - nowRaw;
    if (delta > static_cast<uint64_t>(horizon.count()))
    {
        return SliceTier::Disk;
    }
    return ctx.currentTier;
}

void PressureSpillPolicy::observe(Timestamp now, Timestamp globalWatermark) noexcept
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

/// "reactive" => pure-pressure (no predictor). "predictive" => predictor-refined. Same class, both names retained.
SpillPolicyRegistryReturnType SpillPolicyGeneratedRegistrar::RegisterREACTIVESpillPolicy(SpillPolicyRegistryArguments args)
{
    return std::make_unique<PressureSpillPolicy>(args.highMemoryBound);
}

SpillPolicyRegistryReturnType SpillPolicyGeneratedRegistrar::RegisterPREDICTIVESpillPolicy(SpillPolicyRegistryArguments args)
{
    return std::make_unique<PressureSpillPolicy>(args.highMemoryBound, args.horizon, args.predictorName);
}

/// "always" => evict every slice on every GC tick, whatever the pressure. A zero bound already means
/// exactly that (sample() is clamped to [0, 1], so `pressure < 0.0` is never true), so this is the same
/// class with the configured bound ignored rather than a policy of its own.
///
/// What the evicted slice costs is left to the orthogonal knobs: `compression.enabled` decides whether
/// its bytes are compressed and `spill.backend` decides whether they leave RAM. always + compression +
/// local-file is therefore the always-compress-and-spill baseline, and dropping either knob gives the
/// always-spill-raw or always-compress-in-RAM variant of it.
SpillPolicyRegistryReturnType SpillPolicyGeneratedRegistrar::RegisterALWAYSSpillPolicy(SpillPolicyRegistryArguments)
{
    return std::make_unique<PressureSpillPolicy>(0.0);
}

}
