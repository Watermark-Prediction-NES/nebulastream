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

#include <Experiment.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <Noise/NoiseModel.hpp>
#include <Trace/Trace.hpp>
#include <Trace/TraceSource.hpp>

namespace NES
{

Experiment makeCleanExperiment(std::string name, const TraceSource& source, size_t observePrefix, std::vector<uint64_t> horizons)
{
    auto trace = source.generate();
    return Experiment{
        .name = std::move(name),
        .observed = trace,
        .truth = std::move(trace),
        .observePrefix = observePrefix,
        .horizons = std::move(horizons)};
}

Experiment makeNoisyExperiment(
    std::string name, const TraceSource& source, const NoiseModel& noise, size_t observePrefix, std::vector<uint64_t> horizons)
{
    auto clean = source.generate();
    auto noisy = noise.apply(clean);
    return Experiment{
        .name = std::move(name),
        .observed = std::move(noisy),
        .truth = std::move(clean),
        .observePrefix = observePrefix,
        .horizons = std::move(horizons)};
}

}
