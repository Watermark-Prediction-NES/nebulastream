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
#include <Time/Timestamp.hpp>
#include <Watermark/WatermarkPredictor.hpp>

namespace NES
{

/// Tracks an exponentially weighted moving average of the watermark
/// advancement rate dW/dt_wall. Needs >=2 observations before predicting.
class EwmaWatermarkPredictor final : public WatermarkPredictor
{
public:
    /// @param alpha smoothing factor in (0, 1]; higher = react faster to new rates.
    explicit EwmaWatermarkPredictor(double alpha = 0.3);

    void observe(Timestamp watermarkTs, Timestamp wallClock) override;

    [[nodiscard]] Timestamp predictWallClock(Timestamp target) const override;

private:
    double smoothingFactor;
    double rateEstimate{0.0};
    bool hasRateEstimate{false};
    bool hasFirstObservation{false};
    Timestamp lastWatermark{0};
    Timestamp lastWallClock{0};
};

}
