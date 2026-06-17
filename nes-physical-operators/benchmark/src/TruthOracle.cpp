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

#include <TruthOracle.hpp>

#include <cstddef>
#include <optional>
#include <Time/Timestamp.hpp>
#include <Trace/Trace.hpp>

namespace NES
{

std::optional<Timestamp> trueWallClockForTarget(const WatermarkTrace& trace, Timestamp target)
{
    for (size_t i = 1; i < trace.size(); ++i)
    {
        if (trace[i].watermarkTs >= target && trace[i - 1].watermarkTs < target)
        {
            const auto prevW = static_cast<double>(trace[i - 1].watermarkTs.getRawValue());
            const auto curW = static_cast<double>(trace[i].watermarkTs.getRawValue());
            const auto prevT = static_cast<double>(trace[i - 1].wallClock.getRawValue());
            const auto curT = static_cast<double>(trace[i].wallClock.getRawValue());
            const double frac = (static_cast<double>(target.getRawValue()) - prevW) / (curW - prevW);
            const double tw = prevT + (frac * (curT - prevT));
            return Timestamp{static_cast<Timestamp::Underlying>(tw)};
        }
    }
    return std::nullopt;
}

}
