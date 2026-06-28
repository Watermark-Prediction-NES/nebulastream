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

#include <Watermark/MlpWatermarkPredictor.hpp>

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

namespace
{
/// Lower bound for the extrapolation rate; avoids division by zero on a stalled stream.
constexpr double MIN_RATE = 1e-12;
/// Numerical floor on the normalization scale.
constexpr double NORM_EPS = 1e-8;
/// Adam hyper-parameters (standard defaults; not exposed -- learningRate is the knob that matters).
constexpr double ADAM_B1 = 0.9;
constexpr double ADAM_B2 = 0.999;
constexpr double ADAM_EPS = 1e-8;
/// Number of engineered features appended after the rate window (short EWMA, long EWMA, dt).
constexpr int EXTRA_FEATURES = 3;
/// Symmetric clamp on the dimensionless dt feature, so an extreme inter-arrival gap stays bounded.
constexpr double DT_FEATURE_CLAMP = 10.0;
}

MlpWatermarkPredictor::MlpWatermarkPredictor() : MlpWatermarkPredictor{Config{}}
{
}

MlpWatermarkPredictor::MlpWatermarkPredictor(Config cfg) : cfg{std::move(cfg)}
{
    INVARIANT(this->cfg.windowSize > 0, "MLP windowSize must be > 0, got {}", this->cfg.windowSize);
    INVARIANT(this->cfg.hiddenSize > 0, "MLP hiddenSize must be > 0, got {}", this->cfg.hiddenSize);
    INVARIANT(this->cfg.learningRate > 0.0, "MLP learningRate must be > 0, got {}", this->cfg.learningRate);
    INVARIANT(this->cfg.warmup >= 0, "MLP warmup must be >= 0, got {}", this->cfg.warmup);
    INVARIANT(this->cfg.ewmaAlpha > 0.0 && this->cfg.ewmaAlpha <= 1.0, "MLP ewmaAlpha must be in (0, 1], got {}", this->cfg.ewmaAlpha);
    INVARIANT(
        this->cfg.longEwmaAlpha > 0.0 && this->cfg.longEwmaAlpha <= 1.0,
        "MLP longEwmaAlpha must be in (0, 1], got {}",
        this->cfg.longEwmaAlpha);
    INVARIANT(this->cfg.normBeta > 0.0 && this->cfg.normBeta <= 1.0, "MLP normBeta must be in (0, 1], got {}", this->cfg.normBeta);
    INVARIANT(this->cfg.gradClip > 0.0, "MLP gradClip must be > 0, got {}", this->cfg.gradClip);
    INVARIANT(this->cfg.huberDelta > 0.0, "MLP huberDelta must be > 0, got {}", this->cfg.huberDelta);
    inputDim = this->cfg.windowSize + EXTRA_FEATURES;
}

void MlpWatermarkPredictor::initNetwork()
{
    const auto hidden = static_cast<std::size_t>(cfg.hiddenSize);
    const auto inDim = static_cast<std::size_t>(inputDim);

    std::mt19937_64 rng{cfg.seed};
    /// Xavier-style uniform init keeps the initial activations in tanh's linear region.
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
    w2.resize(hidden);
    for (auto& weight : w2)
    {
        weight = outputDist(rng);
    }
    b2 = 0.0;

    mW1.assign(w1.size(), 0.0);
    vW1.assign(w1.size(), 0.0);
    mB1.assign(hidden, 0.0);
    vB1.assign(hidden, 0.0);
    mW2.assign(hidden, 0.0);
    vW2.assign(hidden, 0.0);
    mB2 = 0.0;
    vB2 = 0.0;
    adamStep = 0;

    prevFeatures.assign(inDim, 0.0);
}

double MlpWatermarkPredictor::normalize(const double rate) const
{
    return (rate - normMean) / std::sqrt(normVar + NORM_EPS);
}

double MlpWatermarkPredictor::denormalize(const double normalized) const
{
    return (normalized * std::sqrt(normVar + NORM_EPS)) + normMean;
}

std::vector<double> MlpWatermarkPredictor::buildFeatures() const
{
    std::vector<double> features(static_cast<std::size_t>(inputDim), 0.0);
    /// Most-recent-first normalized rates; missing entries (early on) stay at the mean (0).
    const std::size_t have = rateWindow.size();
    for (int k = 0; k < cfg.windowSize; ++k)
    {
        if (static_cast<std::size_t>(k) < have)
        {
            features[static_cast<std::size_t>(k)] = normalize(rateWindow[have - 1 - static_cast<std::size_t>(k)]);
        }
    }
    features[static_cast<std::size_t>(cfg.windowSize)] = normalize(shortEwma);
    features[static_cast<std::size_t>(cfg.windowSize) + 1] = normalize(longEwma);
    features[static_cast<std::size_t>(cfg.windowSize) + 2] = dtFeature;
    return features;
}

double MlpWatermarkPredictor::forward(const std::vector<double>& features, std::vector<double>& hiddenOut) const
{
    const auto hidden = static_cast<std::size_t>(cfg.hiddenSize);
    const auto inDim = static_cast<std::size_t>(inputDim);
    hiddenOut.resize(hidden);
    double out = b2;
    for (std::size_t j = 0; j < hidden; ++j)
    {
        double preActivation = b1[j];
        const std::size_t base = j * inDim;
        for (std::size_t k = 0; k < inDim; ++k)
        {
            preActivation += w1[base + k] * features[k];
        }
        const double activation = std::tanh(preActivation);
        hiddenOut[j] = activation;
        out += w2[j] * activation;
    }
    return out;
}

void MlpWatermarkPredictor::trainStep(const double targetNorm)
{
    const auto hidden = static_cast<std::size_t>(cfg.hiddenSize);
    const auto inDim = static_cast<std::size_t>(inputDim);

    /// Recompute the forward pass on the features that produced the cached forecast so we have
    /// the hidden activations for backprop (cheap; this is the comparison-tier predictor).
    std::vector<double> hiddenAct;
    const double out = forward(prevFeatures, hiddenAct);

    /// Huber gradient on the residual: linear in the trust region, clipped outside it -- so a
    /// burst/straggler rate spike cannot yank the weights.
    const double err = out - targetNorm;
    const double delta = cfg.huberDelta;
    const double lossGrad = (std::abs(err) <= delta) ? err : (delta * ((err > 0.0) ? 1.0 : -1.0));

    std::vector<double> gW1(w1.size(), 0.0);
    std::vector<double> gB1(hidden, 0.0);
    std::vector<double> gW2(hidden, 0.0);
    const double gB2 = lossGrad;
    for (std::size_t j = 0; j < hidden; ++j)
    {
        gW2[j] = lossGrad * hiddenAct[j];
        /// Backprop through tanh: da/dz = 1 - a^2.
        const double hiddenGrad = lossGrad * w2[j] * (1.0 - (hiddenAct[j] * hiddenAct[j]));
        gB1[j] = hiddenGrad;
        const std::size_t base = j * inDim;
        for (std::size_t k = 0; k < inDim; ++k)
        {
            gW1[base + k] = hiddenGrad * prevFeatures[k];
        }
    }

    /// Global gradient-norm clipping for stable online updates on volatile streams.
    double sumSquares = gB2 * gB2;
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
        adam(w2[j], mW2[j], vW2[j], gW2[j]);
    }
    adam(b2, mB2, vB2, gB2);
}

void MlpWatermarkPredictor::observe(const Timestamp watermarkTs, const Timestamp wallClock)
{
    if (!initialized)
    {
        lastWatermark = watermarkTs;
        lastWallClock = wallClock;
        initialized = true;
        initNetwork();
        return;
    }

    /// Require dt > 0; reject duplicate / out-of-order wall-clock samples (as EWMA/Kalman do).
    if (wallClock <= lastWallClock)
    {
        return;
    }
    const double watermarkDelta = static_cast<double>(watermarkTs.getRawValue()) - static_cast<double>(lastWatermark.getRawValue());
    const double wallClockDelta = static_cast<double>(wallClock.getRawValue()) - static_cast<double>(lastWallClock.getRawValue());
    const double rate = watermarkDelta / wallClockDelta;

    /// Self-supervised step: the realized rate is the label for the forecast made last step.
    /// Normalize with the pre-update stats so the target matches the scale the forecast used.
    if (hasForecast)
    {
        trainStep(normalize(rate));
    }

    /// Update running normalization (EWMA mean/variance of the rate).
    if (!normInit)
    {
        normMean = rate;
        normVar = 1.0;
        normInit = true;
    }
    else
    {
        const double delta = rate - normMean;
        normMean += cfg.normBeta * delta;
        normVar = (1.0 - cfg.normBeta) * (normVar + (cfg.normBeta * delta * delta));
    }

    /// Update feature state.
    if (count == 0)
    {
        shortEwma = rate;
        longEwma = rate;
        ewmaDt = wallClockDelta;
    }
    else
    {
        shortEwma += cfg.ewmaAlpha * (rate - shortEwma);
        longEwma += cfg.longEwmaAlpha * (rate - longEwma);
        ewmaDt += cfg.ewmaAlpha * (wallClockDelta - ewmaDt);
    }
    dtFeature = std::clamp((wallClockDelta / std::max(ewmaDt, NORM_EPS)) - 1.0, -DT_FEATURE_CLAMP, DT_FEATURE_CLAMP);

    rateWindow.push_back(rate);
    if (rateWindow.size() > static_cast<std::size_t>(cfg.windowSize))
    {
        rateWindow.pop_front();
    }
    ++count;

    /// Build the features for the next step and cache the forecast read by predictWallClock().
    prevFeatures = buildFeatures();
    std::vector<double> hiddenAct;
    const double outNorm = forward(prevFeatures, hiddenAct);
    forecastRate = denormalize(outNorm);
    hasForecast = true;

    lastWatermark = watermarkTs;
    lastWallClock = wallClock;
}

Timestamp MlpWatermarkPredictor::predictWallClock(const Timestamp target) const
{
    /// No rate samples yet -> insufficient state.
    if (!initialized || count == 0)
    {
        return Timestamp{Timestamp::INVALID_VALUE};
    }

    /// Target already reached on or before the last observation.
    if (target <= lastWatermark)
    {
        return lastWallClock;
    }

    /// Cold start: fall back to the EWMA rate so we are never worse than the EWMA baseline
    /// before the net has trained; afterwards use the learned forecast.
    const double rate = (count < static_cast<std::uint64_t>(cfg.warmup)) ? shortEwma : forecastRate;

    const double gap = static_cast<double>(target.getRawValue()) - static_cast<double>(lastWatermark.getRawValue());
    const double safeRate = std::max(rate, MIN_RATE);
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
