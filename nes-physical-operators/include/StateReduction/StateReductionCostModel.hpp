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
#include <StateReductionTypes.hpp>

namespace NES
{

/// What one stage of a transition costs as a function of size: a fixed part that does not depend on how
/// much data is moved (a syscall, a file open, setting up a compression context) and a part that scales
/// with the bytes. Two points are enough to fit it, which is why start-up calibration can be cheap.
struct StageCost
{
    double fixedNanos{0.0};
    double nanosPerByte{0.0};

    [[nodiscard]] double nanosFor(const uint64_t bytes) const { return fixedNanos + (nanosPerByte * static_cast<double>(bytes)); }

    /// Moves the whole curve a `weight` fraction of the way toward being `correction` times steeper.
    /// One timing cannot separate the fixed term from the scaling term, so the honest response to
    /// "we were off by 1.4x" is to scale both rather than pretend to re-fit them independently.
    void applyCorrection(double correction, double weight);
};

/// The measured cost of every stage a decision can involve, plus how well the data compresses.
/// Produced by StateReductionCalibrator at operator start-up and refined at runtime by observations.
struct StateReductionCostModel
{
    StageCost compress;
    StageCost decompress;
    StageCost spillWrite;
    StageCost spillRead;
    /// storedBytes / rawBytes, in (0, 1]. 1.0 means compression buys nothing.
    double compressionRatio{1.0};

    /// Total time to reduce `bytes` of state and get it back again. This is the number that has to fit
    /// in the window the watermark predictor gives us: reducing state we cannot restore in time just
    /// moves the stall from memory pressure to the probe.
    [[nodiscard]] double roundTripNanos(StateDecision decision, uint64_t bytes) const;

    /// Time to move `bytes` of resident state out, without the restore.
    [[nodiscard]] double reduceNanos(StateDecision decision, uint64_t bytes) const;

    /// Time to bring `storedBytes` of reduced state back.
    [[nodiscard]] double restoreNanos(StateDecision decision, uint64_t storedBytes) const;
};

}
