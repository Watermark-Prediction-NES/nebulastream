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

#include <optional>
#include <Time/Timestamp.hpp>
#include <Trace/Trace.hpp>

namespace NES
{

/// Ground-truth wall-clock at which the trace's watermark first reaches or crosses `target`.
/// Linearly interpolates between the two bracketing samples, so it works for any monotonic
/// piecewise-linear watermark trajectory. Returns nullopt if the trace never reaches `target`.
[[nodiscard]] std::optional<Timestamp> trueWallClockForTarget(const WatermarkTrace& trace, Timestamp target);

}
