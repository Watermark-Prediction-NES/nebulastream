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

namespace NES
{

/// A single (watermark, wall-clock) observation point in a recorded or synthesised stream.
struct TraceSample
{
    Timestamp watermarkTs;
    Timestamp wallClock;
};

/// A trace is just a vector of samples in observation order. Data-oriented on purpose so future
/// sources (CSV replay, recorded production traces, NES end-to-end captures) produce the same
/// shape without an inheritance hierarchy on the data side.
using WatermarkTrace = std::vector<TraceSample>;

/// One phase of a piecewise-constant trace: `ticks` samples spaced by `eventStep` in event-time
/// and `wallStep` in wall-clock.
struct Phase
{
    uint64_t ticks;
    uint64_t eventStep;
    uint64_t wallStep;
};

}
