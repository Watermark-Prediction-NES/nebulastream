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

#include <Watermark/KalmanWatermarkPredictor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <Time/Timestamp.hpp>

namespace NES
{

/// Mapping from the textbook constant-velocity Kalman filter to our variables:
///   x       = [watermarkEstimate, rateEstimate]                     (state vector)
///   P       = [[varWatermark, covWatermarkRate],                    (state covariance)
///              [covWatermarkRate, varRate]]
///   F       = [[1, wallClockDelta], [0, 1]]                         (state transition)
///   Q       = diag(processNoiseWatermark, processNoiseRate)         (process noise)
///   z       = observation (the new watermark)                       (measurement)
///   H       = [1, 0]   (we only observe position, not rate)         (measurement matrix)
///   R       = observationNoise                                      (measurement noise)
///   y       = innovation                                            (measurement residual)
///   S       = innovationVariance                                    (innovation covariance)
///   K       = [gainWatermark, gainRate]                             (Kalman gain)

namespace
{
/// Lower bound for the rate when extrapolating. Avoids division by zero when the
/// filter's rate estimate transiently collapses to zero (flat or stalled stream).
constexpr double MIN_RATE = 1e-12;
}

KalmanWatermarkPredictor::KalmanWatermarkPredictor() : KalmanWatermarkPredictor{Config{}}
{
}

KalmanWatermarkPredictor::KalmanWatermarkPredictor(Config cfg) : cfg{std::move(cfg)}
{
}

void KalmanWatermarkPredictor::observe(const Timestamp watermarkTs, const Timestamp wallClock)
{
    const auto observation = static_cast<double>(watermarkTs.getRawValue());

    /// Initialise: seed x from the first measurement (watermarkEstimate = observation), set
    /// P[0,0] = R (varWatermark = observationNoise) and P[1,1] large (varRate = initialRateVariance)
    /// to express maximum prior uncertainty in the unobserved rate component so the next updates
    /// can move it freely.
    if (!initialized)
    {
        watermarkEstimate = observation;
        rateEstimate = 0.0;
        varWatermark = cfg.observationNoise;
        covWatermarkRate = 0.0;
        varRate = cfg.initialRateVariance;
        lastWallClock = wallClock;
        initialized = true;
        return;
    }

    /// Require dt > 0 for F to be well-defined; reject duplicate / out-of-order wall-clock samples.
    if (wallClock <= lastWallClock)
    {
        return;
    }
    const auto wallClockDelta = static_cast<double>(wallClock.getRawValue()) - static_cast<double>(lastWallClock.getRawValue());

    /// Predict state: x_pred = F x with F = [[1, dt], [0, 1]], i.e.
    /// predictedWatermark = watermarkEstimate + dt * rateEstimate; rate is unchanged under the
    /// constant-velocity assumption.
    const auto predictedWatermark = watermarkEstimate + (wallClockDelta * rateEstimate);
    const auto predictedRate = rateEstimate;

    /// Predict covariance: P_pred = F P F^T + Q with diagonal Q = diag(Qw, Qr), expanded into three
    /// scalar updates on our 2x2 covariance. The dt-coupling grows covWatermarkRate -- this is what
    /// later lets a position-only measurement update the rate estimate.
    const auto predictedVarWatermark = varWatermark + (2.0 * wallClockDelta * covWatermarkRate)
        + (wallClockDelta * wallClockDelta * varRate) + cfg.processNoiseWatermark;
    const auto predictedCovWatermarkRate = covWatermarkRate + (wallClockDelta * varRate);
    const auto predictedVarRate = varRate + cfg.processNoiseRate;

    /// Innovation: y = z - H x_pred is how far the new observation lies from the line we just
    /// predicted; S = H P_pred H^T + R is "model uncertainty + measurement noise".
    const auto innovation = observation - predictedWatermark;
    const auto innovationVariance = predictedVarWatermark + cfg.observationNoise;

    /// Kalman gain: K = P_pred H^T / S. With H = [1, 0] this is just the first column of P_pred / S,
    /// so gainWatermark is in [0, 1] and gainRate carries the position-->rate update through
    /// covWatermarkRate -- zero if we don't already have any coupling.
    const auto gainWatermark = predictedVarWatermark / innovationVariance;
    const auto gainRate = predictedCovWatermarkRate / innovationVariance;

    /// State update: x_new = x_pred + K y shifts both watermarkEstimate and rateEstimate toward the
    /// observation, weighted by the gain.
    watermarkEstimate = predictedWatermark + (gainWatermark * innovation);
    rateEstimate = predictedRate + (gainRate * innovation);

    /// Covariance update: P_new = (I - K H) P_pred. Uncertainty shrinks (varWatermark, varRate,
    /// covWatermarkRate all reduce) proportionally to the gain.
    varWatermark = (1.0 - gainWatermark) * predictedVarWatermark;
    covWatermarkRate = (1.0 - gainWatermark) * predictedCovWatermarkRate;
    varRate = predictedVarRate - (gainRate * predictedCovWatermarkRate);

    lastWallClock = wallClock;
}

Timestamp KalmanWatermarkPredictor::predictWallClock(const Timestamp target) const
{
    /// No measurements yet -> posterior is the uninformative prior; caller must treat this as
    /// "I don't know".
    if (!initialized)
    {
        return Timestamp{Timestamp::INVALID_VALUE};
    }

    /// Target lies on the wrong side of the current state estimate: watermarkEstimate has already
    /// crossed target, so the crossing happened on or before lastWallClock.
    const auto targetD = static_cast<double>(target.getRawValue());
    if (targetD <= watermarkEstimate)
    {
        return lastWallClock;
    }

    /// Deterministic first-passage extrapolation along the predicted trajectory (no covariance
    /// propagation needed for a point estimate):
    /// predicted = lastWallClock + (target - watermarkEstimate) / rateEstimate.
    const double gap = targetD - watermarkEstimate;
    const double safeRate = std::max(rateEstimate, MIN_RATE);
    const double predicted = static_cast<double>(lastWallClock.getRawValue()) + (gap / safeRate);

    /// Implementation guard, no Kalman analogue: saturate against overflow.
    /// INVALID_VALUE is reserved for "insufficient state", not "never".
    constexpr auto saturate = static_cast<double>(Timestamp::INVALID_VALUE - 1);
    if (!std::isfinite(predicted) || predicted >= saturate)
    {
        return Timestamp{Timestamp::INVALID_VALUE - 1};
    }
    return Timestamp{static_cast<Timestamp::Underlying>(predicted)};
}

}
