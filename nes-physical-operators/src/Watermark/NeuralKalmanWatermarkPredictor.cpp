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

#include <Watermark/NeuralKalmanWatermarkPredictor.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>
#include <Time/Timestamp.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

/// Constant-velocity Kalman mapping (see KalmanWatermarkPredictor for the full textbook key):
///   x = [watermarkEstimate, rateEstimate]; P = [[varWatermark, covWatermarkRate], [.., varRate]]
///   F = [[1, dt], [0, 1]]; Q = diag(processNoiseWatermark, processNoiseRate); H = [1, 0]; R = observationNoise
///   y = innovation; S = innovationVariance; K_kalman = [gw, gr] = P_pred H^T / S.
/// The only change vs. the plain filter: the gain applied to the *state* is the analytic gain scaled
/// by a learned modulation m = [m_p, m_r] from the gain net. The *covariance* still follows the
/// nominal analytic gain, which keeps P positive and S a meaningful normalizer for the net's input.

namespace
{
/// Lower bound for the extrapolation rate; avoids division by zero on a stalled stream.
constexpr double MIN_RATE = 1e-12;
/// Numerical floor on variances before a square root / division.
constexpr double VAR_EPS = 1e-12;
/// Adam hyper-parameters (standard defaults; not exposed -- learningRate is the knob that matters).
constexpr double ADAM_B1 = 0.9;
constexpr double ADAM_B2 = 0.999;
constexpr double ADAM_EPS = 1e-8;
/// Symmetric clamp on the normalized innovation feature, so a huge spike stays bounded.
constexpr double INNOV_CLAMP = 10.0;
/// Symmetric clamp on the dimensionless dt feature.
constexpr double DT_FEATURE_CLAMP = 10.0;
/// Gain-net input features: z, |z|, fading mean |z|, signed-innovation EWMA, dt feature.
constexpr int NUM_FEATURES = 5;
/// Gain-net outputs: the two gain multipliers m_p, m_r.
constexpr std::size_t OUTPUT_DIM = 2;
}

NeuralKalmanWatermarkPredictor::NeuralKalmanWatermarkPredictor() : NeuralKalmanWatermarkPredictor{Config{}}
{
}

NeuralKalmanWatermarkPredictor::NeuralKalmanWatermarkPredictor(Config cfg) : cfg{std::move(cfg)}
{
    INVARIANT(this->cfg.hiddenSize > 0, "NeuralKalman hiddenSize must be > 0, got {}", this->cfg.hiddenSize);
    INVARIANT(this->cfg.learningRate > 0.0, "NeuralKalman learningRate must be > 0, got {}", this->cfg.learningRate);
    INVARIANT(this->cfg.gradClip > 0.0, "NeuralKalman gradClip must be > 0, got {}", this->cfg.gradClip);
    INVARIANT(this->cfg.huberDelta > 0.0, "NeuralKalman huberDelta must be > 0, got {}", this->cfg.huberDelta);
    INVARIANT(this->cfg.gainModRange > 0.0, "NeuralKalman gainModRange must be > 0, got {}", this->cfg.gainModRange);
    INVARIANT(this->cfg.observationNoise > 0.0, "NeuralKalman observationNoise must be > 0, got {}", this->cfg.observationNoise);
    INVARIANT(
        this->cfg.innovStatsForgetting > 0.0 && this->cfg.innovStatsForgetting <= 1.0,
        "NeuralKalman innovStatsForgetting must be in (0, 1], got {}",
        this->cfg.innovStatsForgetting);
    INVARIANT(
        this->cfg.signEwmaAlpha > 0.0 && this->cfg.signEwmaAlpha <= 1.0,
        "NeuralKalman signEwmaAlpha must be in (0, 1], got {}",
        this->cfg.signEwmaAlpha);
    INVARIANT(
        this->cfg.dtEwmaAlpha > 0.0 && this->cfg.dtEwmaAlpha <= 1.0,
        "NeuralKalman dtEwmaAlpha must be in (0, 1], got {}",
        this->cfg.dtEwmaAlpha);
    inputDim = NUM_FEATURES;
}

void NeuralKalmanWatermarkPredictor::initNetwork()
{
    const auto hidden = static_cast<std::size_t>(cfg.hiddenSize);
    const auto inDim = static_cast<std::size_t>(inputDim);

    std::mt19937_64 rng{cfg.seed};
    /// Xavier-style uniform init keeps initial hidden activations in tanh's linear region; the small
    /// output weights (and zero output bias) make the initial gain modulation m ~= 1, i.e. plain Kalman.
    const double inputScale = std::sqrt(1.0 / static_cast<double>(inputDim));
    const double outputScale = std::sqrt(1.0 / static_cast<double>(cfg.hiddenSize));
    std::uniform_real_distribution<double> inputDist{-inputScale, inputScale};
    std::uniform_real_distribution<double> outputDist{-outputScale, outputScale};

    w1.resize(hidden * inDim);
    for (auto& weight : w1)
    {
        weight = inputDist(rng);
    }
    b1.assign(hidden, 0.0);
    w2.resize(OUTPUT_DIM * hidden);
    for (auto& weight : w2)
    {
        weight = outputDist(rng);
    }
    b2.assign(OUTPUT_DIM, 0.0);

    mW1.assign(w1.size(), 0.0);
    vW1.assign(w1.size(), 0.0);
    mB1.assign(hidden, 0.0);
    vB1.assign(hidden, 0.0);
    mW2.assign(w2.size(), 0.0);
    vW2.assign(w2.size(), 0.0);
    mB2.assign(OUTPUT_DIM, 0.0);
    vB2.assign(OUTPUT_DIM, 0.0);
    adamStep = 0;

    prevFeatures.assign(inDim, 0.0);
    prevHidden.assign(hidden, 0.0);
}

void NeuralKalmanWatermarkPredictor::forward(
    const std::vector<double>& features, std::vector<double>& hiddenOut, double& out0, double& out1) const
{
    const auto hidden = static_cast<std::size_t>(cfg.hiddenSize);
    const auto inDim = static_cast<std::size_t>(inputDim);
    hiddenOut.resize(hidden);
    for (std::size_t j = 0; j < hidden; ++j)
    {
        double preActivation = b1[j];
        const std::size_t base = j * inDim;
        for (std::size_t k = 0; k < inDim; ++k)
        {
            preActivation += w1[base + k] * features[k];
        }
        hiddenOut[j] = std::tanh(preActivation);
    }
    out0 = b2[0];
    out1 = b2[1];
    for (std::size_t j = 0; j < hidden; ++j)
    {
        out0 += w2[j] * hiddenOut[j];
        out1 += w2[hidden + j] * hiddenOut[j];
    }
}

void NeuralKalmanWatermarkPredictor::trainStep(const double residual, const double standardDeviation, const double dtNext)
{
    const auto hidden = static_cast<std::size_t>(cfg.hiddenSize);
    const auto inDim = static_cast<std::size_t>(inputDim);
    const double range = cfg.gainModRange;

    /// Huber gradient on the *normalized* residual (e / sd), so huberDelta is in sigma units and a
    /// genuine straggler/burst spike is clipped to a bounded influence. sd is treated as a constant.
    const double normResidual = residual / standardDeviation;
    const double delta = cfg.huberDelta;
    const double residualSign = (normResidual > 0.0) ? 1.0 : -1.0;
    const double huberPrime = (std::abs(normResidual) <= delta) ? normResidual : (delta * residualSign);
    const double lossGradResidual = huberPrime / standardDeviation;

    /// residual = obs - (p_t + dtNext * r_t); p_t / r_t carry the modulation m_p / m_r through the
    /// step-t gain: dp_t/dm_p = gw_t * y_t, dr_t/dm_r = gr_t * y_t.
    const double dLossDmp = lossGradResidual * (-1.0) * prevKalmanGainWatermark * prevInnovation;
    const double dLossDmr = lossGradResidual * (-dtNext) * prevKalmanGainRate * prevInnovation;

    /// Through m = range * sigmoid(out): dm/dout = m * (1 - m/range).
    const double dmpDout = prevGainModWatermark * (1.0 - (prevGainModWatermark / range));
    const double dmrDout = prevGainModRate * (1.0 - (prevGainModRate / range));
    const double gOut0 = dLossDmp * dmpDout;
    const double gOut1 = dLossDmr * dmrDout;

    std::vector<double> gW1(w1.size(), 0.0);
    std::vector<double> gB1(hidden, 0.0);
    std::vector<double> gW2(w2.size(), 0.0);
    std::vector<double> gB2(OUTPUT_DIM, 0.0);
    gB2[0] = gOut0;
    gB2[1] = gOut1;
    for (std::size_t j = 0; j < hidden; ++j)
    {
        const double activation = prevHidden[j];
        gW2[j] = gOut0 * activation;
        gW2[hidden + j] = gOut1 * activation;
        /// Backprop through tanh: da/dz = 1 - a^2.
        const double hiddenGrad = ((gOut0 * w2[j]) + (gOut1 * w2[hidden + j])) * (1.0 - (activation * activation));
        gB1[j] = hiddenGrad;
        const std::size_t base = j * inDim;
        for (std::size_t k = 0; k < inDim; ++k)
        {
            gW1[base + k] = hiddenGrad * prevFeatures[k];
        }
    }

    /// Global gradient-norm clipping for stable online updates on volatile streams.
    double sumSquares = 0.0;
    for (const double grad : gW1)
    {
        sumSquares += grad * grad;
    }
    for (const double grad : gB1)
    {
        sumSquares += grad * grad;
    }
    for (const double grad : gW2)
    {
        sumSquares += grad * grad;
    }
    for (const double grad : gB2)
    {
        sumSquares += grad * grad;
    }
    const double norm = std::sqrt(sumSquares);
    double clipScale = 1.0;
    if (norm > cfg.gradClip && norm > 0.0)
    {
        clipScale = cfg.gradClip / norm;
    }

    ++adamStep;
    const double biasCorr1 = 1.0 - std::pow(ADAM_B1, static_cast<double>(adamStep));
    const double biasCorr2 = 1.0 - std::pow(ADAM_B2, static_cast<double>(adamStep));
    const double learningRate = cfg.learningRate;
    const auto adam = [&](double& param, double& moment1, double& moment2, const double grad)
    {
        const double clippedGrad = grad * clipScale;
        moment1 = (ADAM_B1 * moment1) + ((1.0 - ADAM_B1) * clippedGrad);
        moment2 = (ADAM_B2 * moment2) + ((1.0 - ADAM_B2) * clippedGrad * clippedGrad);
        const double moment1Hat = moment1 / biasCorr1;
        const double moment2Hat = moment2 / biasCorr2;
        param -= learningRate * moment1Hat / (std::sqrt(moment2Hat) + ADAM_EPS);
    };

    for (std::size_t i = 0; i < w1.size(); ++i)
    {
        adam(w1[i], mW1[i], vW1[i], gW1[i]);
    }
    for (std::size_t j = 0; j < hidden; ++j)
    {
        adam(b1[j], mB1[j], vB1[j], gB1[j]);
    }
    for (std::size_t i = 0; i < w2.size(); ++i)
    {
        adam(w2[i], mW2[i], vW2[i], gW2[i]);
    }
    for (std::size_t outputIdx = 0; outputIdx < OUTPUT_DIM; ++outputIdx)
    {
        adam(b2[outputIdx], mB2[outputIdx], vB2[outputIdx], gB2[outputIdx]);
    }
}

void NeuralKalmanWatermarkPredictor::observe(const Timestamp watermarkTs, const Timestamp wallClock)
{
    const auto observation = static_cast<double>(watermarkTs.getRawValue());

    /// Initialise exactly like KalmanWatermarkPredictor: seed x from the first measurement, P[0,0] = R,
    /// P[1,1] large ("rate unknown"). The gain net is set up so its modulation starts at ~1.
    if (!initialized)
    {
        watermarkEstimate = observation;
        rateEstimate = 0.0;
        varWatermark = cfg.observationNoise;
        covWatermarkRate = 0.0;
        varRate = cfg.initialRateVariance;
        lastWallClock = wallClock;
        initialized = true;
        initNetwork();
        return;
    }

    /// Require dt > 0 for F to be well-defined; reject duplicate / out-of-order wall-clock samples.
    if (wallClock <= lastWallClock)
    {
        return;
    }
    const auto wallClockDelta = static_cast<double>(wallClock.getRawValue()) - static_cast<double>(lastWallClock.getRawValue());

    /// Predict state and covariance (identical to the plain constant-velocity filter).
    const auto predictedWatermark = watermarkEstimate + (wallClockDelta * rateEstimate);
    const auto predictedRate = rateEstimate;
    const auto predictedVarWatermark = varWatermark + (2.0 * wallClockDelta * covWatermarkRate)
        + (wallClockDelta * wallClockDelta * varRate) + cfg.processNoiseWatermark;
    const auto predictedCovWatermarkRate = covWatermarkRate + (wallClockDelta * varRate);
    const auto predictedVarRate = varRate + cfg.processNoiseRate;

    /// Innovation and its (positive) standard deviation; zScore is the scale-free surprise the net reads.
    const auto innovation = observation - predictedWatermark;
    const auto innovationVariance = predictedVarWatermark + cfg.observationNoise;
    const double standardDeviation = std::sqrt(std::max(innovationVariance, VAR_EPS));
    const double zScore = std::clamp(innovation / standardDeviation, -INNOV_CLAMP, INNOV_CLAMP);

    /// Self-supervised step: this innovation is the one-step prediction error of the gain the net
    /// produced last step, so train that step before we build the new features / overwrite the cache.
    if (hasCache)
    {
        trainStep(innovation, standardDeviation, wallClockDelta);
    }

    /// Build the gain-net features: current surprise + a summary of the recent past (all scale-free).
    const double absZ = std::abs(zScore);
    const double pastFadingAbsZ = hasFadingAbsZ ? fadingAbsZ : absZ;
    const double pastEwmaDt = hasEwmaDt ? ewmaDt : wallClockDelta;
    const double dtFeature = std::clamp((wallClockDelta / std::max(pastEwmaDt, VAR_EPS)) - 1.0, -DT_FEATURE_CLAMP, DT_FEATURE_CLAMP);
    std::vector<double> features{zScore, absZ, pastFadingAbsZ, signEwma, dtFeature};

    /// Forward pass -> gain modulation m = range * sigmoid(out) in (0, range), ~1 when untrained.
    std::vector<double> hidden;
    double out0 = 0.0;
    double out1 = 0.0;
    forward(features, hidden, out0, out1);
    const double gainModWatermark = cfg.gainModRange / (1.0 + std::exp(-out0));
    const double gainModRate = cfg.gainModRange / (1.0 + std::exp(-out1));

    /// Analytic Kalman gain K = P_pred H^T / S.
    const auto kalmanGainWatermark = predictedVarWatermark / innovationVariance;
    const auto kalmanGainRate = predictedCovWatermarkRate / innovationVariance;

    /// State update uses the learned (modulated) gain; covariance update uses the nominal analytic
    /// gain so P stays positive and S remains a meaningful normalizer next step.
    watermarkEstimate = predictedWatermark + (gainModWatermark * kalmanGainWatermark * innovation);
    rateEstimate = predictedRate + (gainModRate * kalmanGainRate * innovation);
    varWatermark = (1.0 - kalmanGainWatermark) * predictedVarWatermark;
    covWatermarkRate = (1.0 - kalmanGainWatermark) * predictedCovWatermarkRate;
    varRate = predictedVarRate - (kalmanGainRate * predictedCovWatermarkRate);

    /// Update the running feature state (after the features for this step were built).
    if (!hasFadingAbsZ)
    {
        fadingAbsZ = absZ;
        hasFadingAbsZ = true;
    }
    else
    {
        fadingAbsZ += cfg.innovStatsForgetting * (absZ - fadingAbsZ);
    }
    double signOfInnovation = 0.0;
    if (innovation > 0.0)
    {
        signOfInnovation = 1.0;
    }
    else if (innovation < 0.0)
    {
        signOfInnovation = -1.0;
    }
    signEwma += cfg.signEwmaAlpha * (signOfInnovation - signEwma);
    if (!hasEwmaDt)
    {
        ewmaDt = wallClockDelta;
        hasEwmaDt = true;
    }
    else
    {
        ewmaDt += cfg.dtEwmaAlpha * (wallClockDelta - ewmaDt);
    }

    /// Cache everything needed to backprop this step's gain when the next observation arrives.
    prevFeatures = std::move(features);
    prevHidden = std::move(hidden);
    prevGainModWatermark = gainModWatermark;
    prevGainModRate = gainModRate;
    prevKalmanGainWatermark = kalmanGainWatermark;
    prevKalmanGainRate = kalmanGainRate;
    prevInnovation = innovation;
    hasCache = true;

    lastWallClock = wallClock;
}

Timestamp NeuralKalmanWatermarkPredictor::predictWallClock(const Timestamp target) const
{
    /// No measurements yet -> insufficient state.
    if (!initialized)
    {
        return Timestamp{Timestamp::INVALID_VALUE};
    }

    /// Target already on or behind the current state estimate -> crossing was on/before lastWallClock.
    const auto targetD = static_cast<double>(target.getRawValue());
    if (targetD <= watermarkEstimate)
    {
        return lastWallClock;
    }

    /// Deterministic first-passage extrapolation along the current state estimate.
    const double gap = targetD - watermarkEstimate;
    const double safeRate = std::max(rateEstimate, MIN_RATE);
    const double predicted = static_cast<double>(lastWallClock.getRawValue()) + (gap / safeRate);

    /// Saturate against overflow; INVALID_VALUE is reserved for "insufficient state".
    constexpr auto saturate = static_cast<double>(Timestamp::INVALID_VALUE - 1);
    if (!std::isfinite(predicted) || predicted >= saturate)
    {
        return Timestamp{Timestamp::INVALID_VALUE - 1};
    }
    return Timestamp{static_cast<Timestamp::Underlying>(predicted)};
}

}
