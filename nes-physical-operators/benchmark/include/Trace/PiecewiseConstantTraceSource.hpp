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
#include <Trace/Trace.hpp>
#include <Trace/TraceSource.hpp>

namespace NES
{

/// Generates a piecewise-constant watermark trace by concatenating phases.
/// Use a single-phase configuration for constant-rate streams, multi-phase for rate-change /
/// regime-switch experiments.
class PiecewiseConstantTraceSource final : public TraceSource
{
public:
    explicit PiecewiseConstantTraceSource(std::vector<Phase> phases, uint64_t startWatermark = 1000, uint64_t startWallClock = 1000);

    [[nodiscard]] WatermarkTrace generate() const override;

private:
    std::vector<Phase> phases;
    uint64_t startWatermark;
    uint64_t startWallClock;
};

}
