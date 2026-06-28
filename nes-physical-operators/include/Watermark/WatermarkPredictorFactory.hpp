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

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <string_view>
#include <Watermark/EwmaWatermarkPredictor.hpp>
#include <Watermark/KalmanWatermarkPredictor.hpp>
#include <Watermark/MlpWatermarkPredictor.hpp>
#include <Watermark/RobustAdaptiveKalmanWatermarkPredictor.hpp>
#include <Watermark/WatermarkPredictor.hpp>

namespace NES
{

/// Builds a watermark predictor by name (case-insensitive). Shared by the spill policy and the slice
/// store's predictive preemptive-create path. "off"/"none" -> nullptr (no prediction); everything
/// else (including unknown names and the empty string) -> EWMA, except the two named Kalman variants.
inline std::unique_ptr<WatermarkPredictor> makeWatermarkPredictor(std::string_view name)
{
    std::string upper{name};
    std::ranges::transform(upper, upper.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (upper == "OFF" or upper == "NONE")
    {
        return nullptr;
    }
    if (upper == "KALMAN")
    {
        return std::make_unique<KalmanWatermarkPredictor>();
    }
    if (upper == "ROBUSTKALMAN")
    {
        return std::make_unique<RobustAdaptiveKalmanWatermarkPredictor>();
    }
    if (upper == "MLP")
    {
        return std::make_unique<MlpWatermarkPredictor>();
    }
    return std::make_unique<EwmaWatermarkPredictor>(0.3);
}

}
