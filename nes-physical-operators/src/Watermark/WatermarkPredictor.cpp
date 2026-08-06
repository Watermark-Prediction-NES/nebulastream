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

#include <Watermark/WatermarkPredictor.hpp>

#include <memory>
#include <utility>
#include <Watermark/EwmaWatermarkPredictor.hpp>
#include <Watermark/KalmanWatermarkPredictor.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

std::unique_ptr<WatermarkPredictor> WatermarkPredictor::create(const WatermarkPredictorType type)
{
    switch (type)
    {
        case WatermarkPredictorType::NONE:
            return nullptr;
        case WatermarkPredictorType::EWMA:
            return std::make_unique<EwmaWatermarkPredictor>();
        case WatermarkPredictorType::KALMAN:
            return std::make_unique<KalmanWatermarkPredictor>();
    }
    std::unreachable();
}

}
