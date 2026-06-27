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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <Noise/NoiseModel.hpp>
#include <Trace/Trace.hpp>
#include <Trace/TraceSource.hpp>

namespace NES
{

/// Self-contained description of one benchmark scenario.
///
/// `observed` is what the predictor sees (jittered/late if a noise model was applied).
/// `truth` is the noiseless trace used to compute ground-truth crossing times.
/// For noise-free experiments the two are the same trace.
struct Experiment
{
    std::string name;
    WatermarkTrace observed;
    WatermarkTrace truth;
    size_t warmup; ///< Samples batch-observed before rolling evaluation begins.
    std::vector<uint64_t> horizons;
};

/// Builds an Experiment from a noise-free trace source. observed == truth.
[[nodiscard]] Experiment makeCleanExperiment(std::string name, const TraceSource& source, size_t warmup, std::vector<uint64_t> horizons);

/// Builds an Experiment by applying a noise model to the trace produced by the source.
/// The clean trace from the source is retained as truth; the noisy trace is what the predictor sees.
[[nodiscard]] Experiment
makeNoisyExperiment(std::string name, const TraceSource& source, const NoiseModel& noise, size_t warmup, std::vector<uint64_t> horizons);

}
