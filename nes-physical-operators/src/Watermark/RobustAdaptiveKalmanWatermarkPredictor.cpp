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

#include <Watermark/RobustAdaptiveKalmanWatermarkPredictor.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <Time/Timestamp.hpp>

namespace NES
{

/// Base filter mapping is identical to KalmanWatermarkPredictor; see that file for the textbook
/// constant-velocity Kalman correspondence (x, P, F, Q, z, H, R, y, S, K). This variant only adds the
/// gate / R-adaptation / regime-detection layer documented in the header.

namespace
{
/// Lower bound for the rate when extrapolating; avoids division by zero on a stalled stream.
constexpr double MIN_RATE = 1e-12;
}

RobustAdaptiveKalmanWatermarkPredictor::RobustAdaptiveKalmanWatermarkPredictor() : RobustAdaptiveKalmanWatermarkPredictor{Config{}}
{
}

RobustAdaptiveKalmanWatermarkPredictor::RobustAdaptiveKalmanWatermarkPredictor(Config cfg) : cfg{std::move(cfg)}
{
}

void RobustAdaptiveKalmanWatermarkPredictor::observe(const Timestamp watermarkTs, const Timestamp wallClock)
{
    const auto observation = static_cast<double>(watermarkTs.getRawValue());

    /// Initialise from the first measurement; R starts at the prior observationNoise.
    if (!initialized)
    {
        watermarkEstimate = observation;
        rateEstimate = 0.0;
        rEstimate = cfg.observationNoise;
        varWatermark = rEstimate;
        covWatermarkRate = 0.0;
        varRate = cfg.initialRateVariance;
        empInnovVar = 0.0;
        hasEmpInnovVar = false;
        signRun = 0;
        lastSign = 0;
        regimeActive = false;
        lastWallClock = wallClock;
        initialized = true;
        return;
    }

    /// Require dt > 0; reject duplicate / out-of-order wall-clock samples.
    if (wallClock <= lastWallClock)
    {
        return;
    }
    const auto wallClockDelta = static_cast<double>(wallClock.getRawValue()) - static_cast<double>(lastWallClock.getRawValue());

    /// (3) Regime-aware process noise: while a rate change is suspected (decided on the previous
    /// step), inflate the rate process noise so the rate can move quickly instead of being dragged.
    const double effProcessNoiseRate = regimeActive ? cfg.processNoiseRate * cfg.regimeQBoost : cfg.processNoiseRate;

    /// Predict state: x_pred = F x with F = [[1, dt], [0, 1]].
    const auto predictedWatermark = watermarkEstimate + (wallClockDelta * rateEstimate);
    const auto predictedRate = rateEstimate;

    /// Predict covariance: P_pred = F P F^T + Q.
    const auto predictedVarWatermark = varWatermark + (2.0 * wallClockDelta * covWatermarkRate)
        + (wallClockDelta * wallClockDelta * varRate) + cfg.processNoiseWatermark;
    const auto predictedCovWatermarkRate = covWatermarkRate + (wallClockDelta * varRate);
    const auto predictedVarRate = varRate + effProcessNoiseRate;

    /// Innovation and its nominal variance S = H P_pred H^T + R.
    const auto innovation = observation - predictedWatermark;
    const auto innovationVariance = predictedVarWatermark + rEstimate;
    const double normalizedInnovation = innovation / std::sqrt(innovationVariance);

    /// (1) WoLF-IMQ gate: weight = (1 + z^2 / c^2)^(-1/2). Suppressed during a known regime change so
    /// the post-change samples that re-train the rate are trusted, not rejected. Disabled if c <= 0.
    double weight = 1.0;
    if (!regimeActive && cfg.wolfThreshold > 0.0)
    {
        const double c2 = cfg.wolfThreshold * cfg.wolfThreshold;
        weight = 1.0 / std::sqrt(1.0 + (normalizedInnovation * normalizedInnovation) / c2);
    }

    /// Robust update: inflate the effective measurement noise to R / w^2 (Prop. 3.1). w = 1 recovers
    /// the plain Kalman step; w -> 0 sends the gain to 0 so an outlier barely moves the state.
    const double weightSq = weight * weight;
    const double effObservationNoise = rEstimate / weightSq;
    const double effInnovationVariance = predictedVarWatermark + effObservationNoise;

    /// Kalman gain with the robust (inflated) innovation variance.
    const auto gainWatermark = predictedVarWatermark / effInnovationVariance;
    const auto gainRate = predictedCovWatermarkRate / effInnovationVariance;

    /// State update: x_new = x_pred + K y.
    watermarkEstimate = predictedWatermark + (gainWatermark * innovation);
    rateEstimate = predictedRate + (gainRate * innovation);

    /// Watermarks are monotonic non-decreasing, so the advancement rate cannot be negative. A sharp
    /// deceleration (e.g. a catch-up burst settling back to a slower rate) feeds a run of negative
    /// innovations that, amplified by the regime Q-boost, would otherwise drive the rate estimate
    /// below zero -- which extrapolates to a saturated "never reaches" prediction. Clamp to the
    /// physical floor so the estimate recovers from 0 instead of from a deep negative excursion.
    /// (Floor is 0, not a positive value: a genuine stall legitimately needs the rate to reach 0.)
    rateEstimate = std::max(rateEstimate, 0.0);

    /// Covariance update: P_new = (I - K H) P_pred.
    varWatermark = (1.0 - gainWatermark) * predictedVarWatermark;
    covWatermarkRate = (1.0 - gainWatermark) * predictedCovWatermarkRate;
    varRate = predictedVarRate - (gainRate * predictedCovWatermarkRate);

    /// (2) Covariance-matching R-adaptation with fading memory: match the empirical innovation
    /// variance to predictedVarWatermark + R and solve for R. Skipped only during a known regime
    /// change (the key "don't inflate R during a rate change" requirement); it runs on every other
    /// sample -- including gated ones -- so that a too-small initial R cannot lock the gate on and
    /// starve the estimate. Covariance matching is inherently non-robust to a sparse straggler tail:
    /// the gate protects the STATE from such spikes, but a spike still blips R for ~1/forgetting
    /// samples. This self-calibrates and is acceptable; it is not a robust R estimator.
    if (cfg.rAdaptForgetting > 0.0 && !regimeActive)
    {
        const double innovationSq = innovation * innovation;
        empInnovVar = hasEmpInnovVar ? ((1.0 - cfg.rAdaptForgetting) * empInnovVar) + (cfg.rAdaptForgetting * innovationSq) : innovationSq;
        hasEmpInnovVar = true;
        rEstimate = std::max(empInnovVar - predictedVarWatermark, cfg.rFloor);
    }

    /// (3) Update the signed-run regime detector for the next step.
    const int sign = normalizedInnovation > cfg.regimeSigma ? 1 : (normalizedInnovation < -cfg.regimeSigma ? -1 : 0);
    if (sign != 0 && sign == lastSign)
    {
        ++signRun;
    }
    else if (sign != 0)
    {
        signRun = 1;
        lastSign = sign;
    }
    else
    {
        signRun = 0;
        lastSign = 0;
    }
    regimeActive = signRun >= cfg.regimeRun;

    lastWallClock = wallClock;
}

Timestamp RobustAdaptiveKalmanWatermarkPredictor::predictWallClock(const Timestamp target) const
{
    /// No measurements yet -> caller must treat this as "I don't know".
    if (!initialized)
    {
        return Timestamp{Timestamp::INVALID_VALUE};
    }

    /// Target already crossed by the current estimate -> crossing was on or before lastWallClock.
    const auto targetD = static_cast<double>(target.getRawValue());
    if (targetD <= watermarkEstimate)
    {
        return lastWallClock;
    }

    /// Deterministic first-passage extrapolation: lastWallClock + (target - watermarkEstimate) / rate.
    const double gap = targetD - watermarkEstimate;
    const double safeRate = std::max(rateEstimate, MIN_RATE);
    const double predicted = static_cast<double>(lastWallClock.getRawValue()) + (gap / safeRate);

    /// Saturate against overflow; INVALID_VALUE is reserved for "insufficient state", not "never".
    constexpr auto saturate = static_cast<double>(Timestamp::INVALID_VALUE - 1);
    if (!std::isfinite(predicted) || predicted >= saturate)
    {
        return Timestamp{Timestamp::INVALID_VALUE - 1};
    }
    return Timestamp{static_cast<Timestamp::Underlying>(predicted)};
}

}
