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
#include <cstdint>
#include <deque>
#include <vector>
#include <Time/Timestamp.hpp>
#include <Watermark/WatermarkPredictor.hpp>

namespace NES
{

/// Online machine-learning watermark predictor: a single-hidden-layer perceptron trained
/// incrementally from the live stream (backprop + Adam), with no pre-training.
///
/// Unlike the EWMA/Kalman predictors -- hand-derived recursive estimators with a fixed
/// linear model -- this learns its weights from data. It is the heavier "what can a learned
/// model extract" comparison rather than a production-lightweight estimator, yet still pure
/// C++23 with zero external dependencies and bounded memory.
///
/// Chosen as a windowed MLP (not a recurrent net) for NES's cloud-edge-sensor / IoT setting
/// with highly volatile data rates: it trains stably online, forgets stale regimes within
/// its window (fast volatility adaptation), and is light enough for edge nodes.
///
/// Design (same scaffolding as the other predictors): forecast the near-future advancement
/// rate r_hat, then extrapolate t* = lastWallClock + (target - lastWatermark) / r_hat. Only
/// the production of r_hat is learned. The network regresses the *normalized residual* of the
/// rate around a slowly-adapting running mean, so even an untrained net predicts ~the recent
/// mean rate -- it is never wildly off, and learns the structured deviations on top.
class MlpWatermarkPredictor final : public WatermarkPredictor
{
public:
    struct Config
    {
        int windowSize{16}; ///< Number of recent per-step rates fed as input features.
        int hiddenSize{16}; ///< Hidden-layer width.
        double learningRate{1e-3}; ///< Adam step size.
        int warmup{8}; ///< Rate samples before trusting the net; falls back to EWMA before.
        double ewmaAlpha{0.3}; ///< Short EWMA (cold-start fallback rate + a feature).
        double longEwmaAlpha{0.05}; ///< Long EWMA (a slower-memory feature).
        double normBeta{0.02}; ///< Adaptation rate of the running input/target normalization.
        double gradClip{1.0}; ///< Global gradient-norm clip; keeps online training stable.
        double huberDelta{1.0}; ///< Huber-loss knee; robust to burst/straggler rate spikes.
        std::uint64_t seed{42}; ///< Deterministic weight initialization (reproducible tests).
    };

    MlpWatermarkPredictor();
    explicit MlpWatermarkPredictor(Config cfg);

    void observe(Timestamp watermarkTs, Timestamp wallClock) override;

    [[nodiscard]] Timestamp predictWallClock(Timestamp target) const override;

    /// Current learned rate forecast (event-time units per wall-clock unit); 0 before any rate.
    [[nodiscard]] double currentRateEstimate() const noexcept { return forecastRate; }

private:
    void initNetwork();
    [[nodiscard]] double normalize(double rate) const;
    [[nodiscard]] double denormalize(double normalized) const;
    [[nodiscard]] std::vector<double> buildFeatures() const;
    /// Forward pass; fills the hidden activations and returns the (normalized) output.
    [[nodiscard]] double forward(const std::vector<double>& features, std::vector<double>& hiddenOut) const;
    void trainStep(double targetNorm);

    Config cfg;
    int inputDim{0}; ///< windowSize + 3 (short EWMA, long EWMA, dt feature).

    /// Network parameters (row-major w1 is hiddenSize x inputDim; w2 is hiddenSize).
    std::vector<double> w1;
    std::vector<double> b1;
    std::vector<double> w2;
    double b2{0.0};

    /// Adam first/second moments, mirroring the parameter layout, plus the step counter.
    std::vector<double> mW1, vW1, mB1, vB1, mW2, vW2;
    double mB2{0.0}, vB2{0.0};
    std::uint64_t adamStep{0};

    /// Feature state.
    std::deque<double> rateWindow; ///< Most-recent-last raw rates, capped at windowSize.
    double shortEwma{0.0}; ///< EWMA(ewmaAlpha) rate -- also the cold-start fallback.
    double longEwma{0.0}; ///< EWMA(longEwmaAlpha) rate.
    double ewmaDt{0.0}; ///< EWMA of the wall-clock step, for the dimensionless dt feature.
    double dtFeature{0.0}; ///< Last (dt / ewmaDt - 1), clamped.

    /// Running normalization (EWMA mean/variance of the rate; slowly adapting).
    double normMean{0.0};
    double normVar{1.0};
    bool normInit{false};

    /// Online-training linkage: the features that produced the cached forecast, and the
    /// cached (denormalized) forecast itself read by predictWallClock().
    std::vector<double> prevFeatures;
    bool hasForecast{false};
    double forecastRate{0.0};

    std::uint64_t count{0}; ///< Number of valid rate samples observed.
    bool initialized{false};
    Timestamp lastWatermark{0};
    Timestamp lastWallClock{0};
};

}
