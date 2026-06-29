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
#include <vector>
#include <Time/Timestamp.hpp>
#include <Watermark/WatermarkPredictor.hpp>

namespace NES
{

/// Online machine-learning watermark predictor: a constant-velocity Kalman filter whose
/// Kalman gain is produced by a small neural network trained online by gradient descent.
///
/// This is the KalmanNet idea -- Revach et al., "KalmanNet: Neural Network Aided Kalman
/// Filtering for Partially Known Dynamics", IEEE Transactions on Signal Processing, 2022
/// (arXiv:2107.10043) -- adapted to be *online* and *lightweight*. KalmanNet replaces the
/// analytically-computed gain of a Kalman filter with a network that learns it from data; the
/// model structure (here the constant-velocity state and the prediction scaffolding) is kept,
/// only the gain is learned. Our adaptation: (1) self-supervised online training -- the next
/// observation's innovation is the label, no pre-training; (2) a feedforward gain net over
/// hand-crafted innovation features instead of an RNN, so there is no backprop-through-time and
/// the only recurrence is the (stable, bounded) Kalman state itself.
///
/// Why this beats the hand-tuned filters in principle: EWMA has a fixed smoothing factor and
/// plain Kalman a fixed gain schedule (set by fixed Q/R); RobustAdaptiveKalman approximates a
/// context-dependent gain with hand-tuned thresholds (outlier gate, regime detector). Here the
/// network *learns* that context dependence from the actual prediction error -- it can down-weight
/// the gain under jitter (trust the model) and up-weight it on a sustained rate change (trust the
/// observation), which is exactly what the hand-tuned heuristics try to do by hand.
///
/// Genuine ML, yet edge-light: the learned gain is the output of a backprop-trained nonlinear net
/// (~tens of weights); per-observe cost is one forward+backward pass; per-predict cost is O(1).
/// Pure C++23, zero external dependencies, bounded memory. The net is initialized so its gain
/// modulation starts at ~1.0, i.e. cold start behaves like the plain Kalman filter and is never
/// worse than that baseline before the net has trained.
class NeuralKalmanWatermarkPredictor final : public WatermarkPredictor
{
public:
    struct Config
    {
        /// Base constant-velocity Kalman -- identical roles to KalmanWatermarkPredictor::Config.
        double processNoiseWatermark{1.0}; ///< Per-step variance kick on the watermark position.
        double processNoiseRate{1e-4}; ///< Per-step variance kick on the advancement rate.
        double observationNoise{1.0}; ///< Variance on each observed watermark.
        double initialRateVariance{1e6}; ///< Starting variance on the rate; large = "rate unknown".

        /// Gain network.
        int hiddenSize{8}; ///< Hidden-layer width of the gain net.
        double learningRate{1e-3}; ///< Adam step size.
        double gradClip{1.0}; ///< Global gradient-norm clip; keeps online training stable.
        double huberDelta{1.0}; ///< Huber-loss knee on the normalized innovation; robust to spikes.
        double gainModRange{2.0}; ///< Gain multiplier m in (0, gainModRange); init ~1 == plain Kalman.
        std::uint64_t seed{42}; ///< Deterministic weight initialization (reproducible tests).

        /// Feature memory.
        double innovStatsForgetting{0.05}; ///< EWMA weight for the running mean of |normalized innovation|.
        double signEwmaAlpha{0.3}; ///< EWMA weight for the signed-innovation regime feature.
        double dtEwmaAlpha{0.3}; ///< EWMA weight for the wall-clock step, for the dimensionless dt feature.
    };

    NeuralKalmanWatermarkPredictor();
    explicit NeuralKalmanWatermarkPredictor(Config cfg);

    void observe(Timestamp watermarkTs, Timestamp wallClock) override;

    [[nodiscard]] Timestamp predictWallClock(Timestamp target) const override;

    [[nodiscard]] double currentRateEstimate() const noexcept { return rateEstimate; }

private:
    void initNetwork();
    /// Gain-net forward pass; fills the hidden activations and the two raw outputs.
    void forward(const std::vector<double>& features, std::vector<double>& hiddenOut, double& out0, double& out1) const;
    /// One-step self-supervised update: backprop the Huber loss of the normalized residual
    /// (e / sd) through the cached step into the gain-net weights, then an Adam step.
    void trainStep(double residual, double standardDeviation, double dtNext);

    Config cfg;
    int inputDim{0}; ///< Number of gain-net input features.

    /// Constant-velocity Kalman state (same layout as KalmanWatermarkPredictor).
    double watermarkEstimate{0.0}; ///< Smoothed watermark estimate (event-time units).
    double rateEstimate{0.0}; ///< Smoothed advancement rate (event-time per wall-clock unit).
    double varWatermark{0.0}; ///< Variance of watermarkEstimate.
    double covWatermarkRate{0.0}; ///< Covariance between watermarkEstimate and rateEstimate.
    double varRate{0.0}; ///< Variance of rateEstimate.

    /// Gain-net parameters: w1 is hiddenSize x inputDim, w2 is OUTPUT_DIM x hiddenSize (row-major).
    std::vector<double> w1, b1, w2, b2;
    /// Adam first/second moments, mirroring the parameter layout, plus the step counter.
    std::vector<double> mW1, vW1, mB1, vB1, mW2, vW2, mB2, vB2;
    std::uint64_t adamStep{0};

    /// Feature state (all scale-free, so no input normalization is needed).
    double fadingAbsZ{0.0}; ///< Fading-memory mean of |normalized innovation| (noise / jitter level).
    bool hasFadingAbsZ{false};
    double signEwma{0.0}; ///< EWMA of sign(innovation) in [-1, 1] (sustained run = rate change).
    double ewmaDt{0.0}; ///< EWMA of the wall-clock step, for the dimensionless dt feature.
    bool hasEwmaDt{false};

    /// Online-training linkage: everything from step t needed to backprop one step later.
    bool hasCache{false};
    std::vector<double> prevFeatures; ///< Features that produced the cached gain.
    std::vector<double> prevHidden; ///< Hidden activations of that forward pass.
    double prevGainModWatermark{1.0}; ///< Applied position gain multiplier m_p.
    double prevGainModRate{1.0}; ///< Applied rate gain multiplier m_r.
    double prevKalmanGainWatermark{0.0}; ///< Analytic position gain gw at step t.
    double prevKalmanGainRate{0.0}; ///< Analytic rate gain gr at step t.
    double prevInnovation{0.0}; ///< Innovation y at step t.

    bool initialized{false};
    Timestamp lastWallClock{0};
};

}
