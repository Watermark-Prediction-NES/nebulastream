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

#include <SliceStore/Spill/PredictiveSpillPolicy.hpp>

#include <chrono>
#include <memory>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Time/Timestamp.hpp>
#include <Watermark/EwmaWatermarkPredictor.hpp>
#include <SpillPolicyRegistry.hpp>

namespace NES
{

PredictiveSpillPolicy::PredictiveSpillPolicy(double lowBound_, double highBound_, std::chrono::milliseconds horizon_) noexcept
    : lowBound(lowBound_), highBound(highBound_), horizon(horizon_), predictor(std::make_unique<EwmaWatermarkPredictor>(0.3))
{
}

SpillDecision PredictiveSpillPolicy::decide(const SliceSpillContext& ctx, double memoryPressure) const
{
    /// Reactive guard: under-pressure cases short-circuit; predictor is consulted only when
    /// memory is constrained.
    if (memoryPressure < highBound)
    {
        return SpillDecision::Keep;
    }

    /// Ask the predictor when the watermark will reach the slice's end. If unknown (cold predictor),
    /// fall back to reactive behaviour and just spill.
    const Timestamp predictedTrigger = predictor->predictWallClock(ctx.sliceEnd);
    if (predictedTrigger.getRawValue() == Timestamp::INVALID_VALUE)
    {
        return SpillDecision::Spill;
    }

    /// If the predicted trigger is further than `horizon` from now, spill. Otherwise keep the slice
    /// resident so that the probe doesn't pay restore latency on an imminent trigger.
    const auto nowRaw = ctx.now.getRawValue();
    const auto predRaw = predictedTrigger.getRawValue();
    if (predRaw <= nowRaw)
    {
        return SpillDecision::Keep;
    }
    const auto delta = predRaw - nowRaw;
    if (delta > static_cast<uint64_t>(horizon.count()))
    {
        return SpillDecision::Spill;
    }
    return SpillDecision::Keep;
}

void PredictiveSpillPolicy::observe(Timestamp now, Timestamp globalWatermark) noexcept
{
    if (globalWatermark.getRawValue() == Timestamp::INVALID_VALUE)
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

SpillPolicyRegistryReturnType SpillPolicyGeneratedRegistrar::RegisterPREDICTIVESpillPolicy(SpillPolicyRegistryArguments args)
{
    return std::make_unique<PredictiveSpillPolicy>(args.lowMemoryBound, args.highMemoryBound, args.horizon);
}

}
