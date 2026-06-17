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

#include <Trace/PiecewiseConstantTraceSource.hpp>

#include <cstdint>
#include <utility>
#include <vector>
#include <Time/Timestamp.hpp>
#include <Trace/Trace.hpp>

namespace NES
{

PiecewiseConstantTraceSource::PiecewiseConstantTraceSource(std::vector<Phase> phases, uint64_t startWatermark, uint64_t startWallClock)
    : phases{std::move(phases)}, startWatermark{startWatermark}, startWallClock{startWallClock}
{
}

WatermarkTrace PiecewiseConstantTraceSource::generate() const
{
    WatermarkTrace trace;
    uint64_t w = startWatermark;
    uint64_t t = startWallClock;
    for (const auto& phase : phases)
    {
        for (uint64_t i = 0; i < phase.ticks; ++i)
        {
            trace.push_back({Timestamp{w}, Timestamp{t}});
            w += phase.eventStep;
            t += phase.wallStep;
        }
    }
    return trace;
}

}
