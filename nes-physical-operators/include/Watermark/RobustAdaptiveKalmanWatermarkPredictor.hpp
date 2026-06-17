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

/// Constant-velocity Kalman filter (same state as KalmanWatermarkPredictor) hardened with three
/// published adaptive/robust mechanisms, each cheap enough to keep the O(1) per-predict cost:
///
///   1. WoLF-IMQ robust gate -- Duran-Martin et al., "Outlier-robust Kalman Filtering through
///      Generalised Bayes", ICML 2024 (arXiv:2405.05646). Down-weights a sample by
///      w = (1 + z^2 / c^2)^(-1/2) on the normalized innovation z, which is equivalent to inflating
///      the measurement noise to R / w^2 (their Prop. 3.1, R^-1 -> w^2 R^-1). Rejects one-off
///      straggler spikes without an iteration or a matrix inversion.
///   2. Covariance-matching R-adaptation -- Myers & Tapley, IEEE TAC 1976 -- with exponential fading
///      memory -- Gao et al., J. Navigation 2015. Tracks the real measurement noise online from the
///      empirical innovation variance, so a fixed R need not be guessed up front.
///   3. Signed-run regime detector -- the state-vs-measurement-outlier split of SWRKF
///      (Zhu et al., IEEE TAES 2022) without the variational-Bayes machinery. A run of same-sign
///      large innovations is read as a genuine rate change (raise process-rate noise, do NOT gate),
///      so a regime shift is not mistaken for measurement noise.
///
/// State = [watermark position, advancement rate]; observation = watermark only; prediction is the
/// same deterministic first-passage extrapolation as KalmanWatermarkPredictor.
class RobustAdaptiveKalmanWatermarkPredictor final : public WatermarkPredictor
{
public:
    struct Config
    {
        /// Base constant-velocity Kalman -- identical roles to KalmanWatermarkPredictor::Config.
        double processNoiseWatermark{1.0}; ///< Per-step variance kick on the watermark position.
        double processNoiseRate{1e-4}; ///< Per-step variance kick on the advancement rate.
        double observationNoise{1.0}; ///< Prior measurement variance; the adapted R starts here.
        double initialRateVariance{1e6}; ///< Starting variance on the rate; large = "rate unknown".

        /// (1) WoLF-IMQ gate. Soft threshold on the normalized innovation z = innovation / sqrt(S);
        /// larger gates less. <= 0 disables gating (plain Kalman update).
        double wolfThreshold{3.0};

        /// (2) Covariance-matching R-adaptation. forgetting in (0, 1] = weight on the newest
        /// residual in the fading-memory innovation-variance estimate. 0 disables (R stays fixed).
        double rAdaptForgetting{0.05};
        double rFloor{1e-6}; ///< Lower clamp on the adapted R; keeps S strictly positive.

        /// (3) Signed-run regime detector. regimeRun consecutive same-sign innovations with
        /// |z| > regimeSigma are read as a rate change: the gate is suppressed and processNoiseRate
        /// is multiplied by regimeQBoost so the rate re-adapts fast.
        double regimeSigma{3.0};
        int regimeRun{3};
        double regimeQBoost{1e4};
    };

    RobustAdaptiveKalmanWatermarkPredictor();
    explicit RobustAdaptiveKalmanWatermarkPredictor(Config cfg);

    void observe(Timestamp watermarkTs, Timestamp wallClock) override;

    [[nodiscard]] Timestamp predictWallClock(Timestamp target) const override;

    [[nodiscard]] double currentRateEstimate() const noexcept { return rateEstimate; }

    [[nodiscard]] double currentObservationNoise() const noexcept { return rEstimate; }

    /// Q/R are read fresh every observe() and (x, P) is a sufficient statistic, so the tuning can be
    /// swapped between samples with no replay. initialRateVariance is read only at init -> a no-op
    /// once initialized.
    void setConfig(Config c) noexcept { cfg = c; }

private:
    Config cfg;

    /// Estimates (event-time units; rate in event-time per wall-clock unit).
    double watermarkEstimate{0.0};
    double rateEstimate{0.0};

    /// Covariance entries (symmetric 2x2 stored as 3 doubles).
    double varWatermark{0.0};
    double covWatermarkRate{0.0};
    double varRate{0.0};

    /// Adaptation state.
    double rEstimate{0.0}; ///< Current (adapted) measurement variance R.
    double empInnovVar{0.0}; ///< Fading-memory estimate of the innovation variance.
    bool hasEmpInnovVar{false};
    int signRun{0}; ///< Consecutive same-sign large normalized innovations.
    int lastSign{0}; ///< Sign of the last large innovation (-1 / 0 / +1).
    bool regimeActive{false}; ///< True while a rate change is suspected (set for the next step).

    bool initialized{false};
    Timestamp lastWallClock{0};
};

}
