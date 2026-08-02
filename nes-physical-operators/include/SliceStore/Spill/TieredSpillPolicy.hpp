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
#include <memory>
#include <string_view>
#include <SliceStore/Spill/SpillPolicy.hpp>
#include <Time/Timestamp.hpp>
#include <Watermark/WatermarkPredictor.hpp>

namespace NES

{

/// Places each slice in a memory tier from how soon the watermark predictor says it will be probed.
/// The further away the probe, the more we are willing to pay to get the slice back:
///
///   time-to-trigger <= promoteHorizon   Resident        (restore it before the probe stalls on it)
///   time-to-trigger <= nearHorizon      Resident        (too soon to be worth evicting)
///   time-to-trigger <= compressRamH     CompressedRam   (smaller, still in RAM, decompress to restore)
///   time-to-trigger <= compressDiskH    CompressedDisk  (out of RAM, less I/O than raw)
///   otherwise                           Disk
///
/// Registered as "tiered". With a cold predictor it degrades to the reactive policy exactly, which is
/// what makes it safe to enable by default.
class TieredSpillPolicy final : public SpillPolicy
{
public:
    TieredSpillPolicy(
        double highBound,
        std::chrono::milliseconds promoteHorizon,
        std::chrono::milliseconds nearHorizon,
        std::chrono::milliseconds compressRamHorizon,
        std::chrono::milliseconds compressDiskHorizon,
        std::string_view predictorName) noexcept;

    [[nodiscard]] SliceTier decide(const SliceSpillContext& ctx, double memoryPressure) const override;

    void observe(Timestamp now, Timestamp globalWatermark) noexcept override;

private:
    double highBound;
    /// Clamped monotone at construction, so the ladder below always reads in order.
    std::chrono::milliseconds promoteHorizon;
    std::chrono::milliseconds nearHorizon;
    std::chrono::milliseconds compressRamHorizon;
    std::chrono::milliseconds compressDiskHorizon;
    std::unique_ptr<WatermarkPredictor> predictor; /// null => degrades to pure-pressure
    Timestamp lastObservedWatermark{Timestamp::INVALID_VALUE};
};

}
