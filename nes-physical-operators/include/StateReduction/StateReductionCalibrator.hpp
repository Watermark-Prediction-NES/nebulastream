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
#include <StateReduction/CompressionAlgorithm.hpp>
#include <StateReduction/SpillStore.hpp>
#include <StateReduction/StateReductionCostModel.hpp>

namespace NES
{

struct CalibrationConfig
{
    /// Geometric ladder of payload sizes. Two points would be enough to fit a line; the ladder spans
    /// three orders of magnitude so the fit is not dominated by whichever end happens to be noisy.
    uint64_t minBytes{4096};
    uint64_t maxBytes{1024 * 1024};
    /// Timings at this scale are noisy, so each size is measured repeatedly and the best run is kept —
    /// the fastest run is the one least contaminated by scheduling and page faults.
    uint32_t repetitions{3};
};

/// Measures what compression and spilling cost on the machine the query is actually running on.
///
/// This is the half of the cost model that cannot be predicted, only measured: an SSD and a spinning
/// disk differ by two orders of magnitude, and so do a debug and a release build of the compressor. It
/// runs once at operator start-up and costs on the order of tens of milliseconds. The other half — how
/// long we have before the state is needed — comes from the WatermarkPredictor at runtime.
///
/// The compression ratio it reports comes from synthetic data, so it is only a starting point;
/// StatePredictor::update replaces it with the real thing within the first few reduced slices.
[[nodiscard]] StateReductionCostModel
calibrateStateReduction(const CompressionAlgorithm& compression, SpillStore& spillStore, const CalibrationConfig& config);

}
