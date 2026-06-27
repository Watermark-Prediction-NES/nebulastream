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

#include <string>
#include <vector>
#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/ScalarOption.hpp>

namespace NES
{

/// Independent slice-store throughput knobs, all sentinel-disabled (0 / "off").
///
/// preemptiveCreateHorizonMs: when build creates slice S, also create slices covering the next N ms
/// in the same critical section. Cuts lock contention from concurrent build tuples that would
/// otherwise each race to create the next slice. With preemptivePredictor == "off" the N ms are
/// event-time (static lookahead, N/slide slices). With a predictor, N ms are WALL-CLOCK: the predictor
/// converts the wall-clock horizon into however many upcoming slices the stream is expected to reach
/// in that window (bounded by preemptiveMaxSlices), so a fast stream warms more slices than a slow one.
///
/// preemptivePredictor: watermark predictor driving the wall-clock horizon (ewma, kalman, robustkalman).
/// "off" = static event-time lookahead. Mirrors the spill subsystem's `predictor` knob.
///
/// preemptiveMaxSlices: hard cap on slices created in one preemptive burst. Stops runaway when the
/// stream's event-time races far ahead of wall-clock (e.g. memory sources), where the wall-clock
/// horizon would otherwise map to an unbounded event-time span. Only used when a predictor is active.
///
/// recyclePoolSize: bounded LIFO of slices retired by probe-side GC. Build pops + resets instead of
/// allocating a fresh slice + hashmaps/paged-vectors.
class SlicePreallocationConfiguration final : public BaseConfiguration
{
public:
    SlicePreallocationConfiguration() = default;
    SlicePreallocationConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { };

    UIntOption preemptiveCreateHorizonMs
        = {"preemptive_create_horizon_ms",
           "0",
           "Build-side lookahead window for preemptive slice creation. 0 disables. When >0, on each "
           "new slice creation the store also creates all slices that would fall within the next N ms. "
           "With preemptive_predictor=off these are event-time ms (N/slide slices); with a predictor "
           "they are wall-clock ms, converted to a slice count by the predictor (capped by preemptive_max_slices)."};

    StringOption preemptivePredictor
        = {"preemptive_predictor",
           "off",
           "Watermark predictor driving the (wall-clock) preemptive-create horizon: off, ewma, kalman, robustkalman. "
           "off = static event-time lookahead. Mirrors the spill subsystem's `predictor` knob."};

    UIntOption preemptiveMaxSlices
        = {"preemptive_max_slices",
           "256",
           "Hard cap on slices created in a single preemptive burst when a predictor is active. Prevents runaway "
           "on fast streams whose event-time races ahead of wall-clock."};

    UIntOption recyclePoolSize
        = {"recycle_pool_size",
           "0",
           "Bounded size of the slice recycle pool. 0 disables. When >0, slices that would be deleted "
           "by probe-side GC are pushed onto a LIFO pool and reused by future build-side allocations "
           "(via Slice::reset). LIFO + drop-oldest on overflow."};

private:
    std::vector<BaseOption*> getOptions() override
    {
        return {&preemptiveCreateHorizonMs, &preemptivePredictor, &preemptiveMaxSlices, &recyclePoolSize};
    }
};

}
