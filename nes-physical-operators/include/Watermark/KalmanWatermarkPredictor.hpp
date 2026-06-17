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

/// Constant-velocity Kalman filter on the watermark trajectory.
/// State = [watermark position, advancement rate]; observation = watermark only.
/// Predicts wall-clock crossing time as lastWallClock + (target - watermarkEstimate) / rateEstimate.
class KalmanWatermarkPredictor final : public WatermarkPredictor
{
public:
    struct Config
    {
        double processNoiseWatermark{1.0}; ///< Per-step variance kick on the watermark position.
        double processNoiseRate{1e-4}; ///< Per-step variance kick on the advancement rate.
        double observationNoise{1.0}; ///< Variance on each observed watermark.
        double initialRateVariance{1e6}; ///< Starting variance on the rate; large = "rate unknown".
    };

    KalmanWatermarkPredictor();
    explicit KalmanWatermarkPredictor(Config cfg);

    void observe(Timestamp watermarkTs, Timestamp wallClock) override;

    [[nodiscard]] Timestamp predictWallClock(Timestamp target) const override;

    [[nodiscard]] double currentRateEstimate() const noexcept { return rateEstimate; }

private:
    Config cfg;

    /// Smoothed estimate of the current watermark (event-time units).
    double watermarkEstimate{0.0};
    /// Smoothed estimate of the watermark advancement rate (event-time units per wall-clock unit).
    double rateEstimate{0.0};

    /// Covariance entries (the 2x2 covariance matrix is symmetric, so we store 3 doubles).
    double varWatermark{0.0}; ///< Variance of watermarkEstimate.
    double covWatermarkRate{0.0}; ///< Covariance between watermarkEstimate and rateEstimate.
    double varRate{0.0}; ///< Variance of rateEstimate.

    bool initialized{false};
    Timestamp lastWallClock{0};
};

}
