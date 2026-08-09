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

#include <memory>
#include <string>
#include <vector>
#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/ScalarOption.hpp>
#include <Configurations/Validation/NonZeroValidation.hpp>

namespace NES
{

/// How a window operator's slice store creates, pools, and hands out its slices.
class SliceStoreConfiguration final : public BaseConfiguration
{
public:
    SliceStoreConfiguration() = default;
    SliceStoreConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { };

    BoolOption enableSliceRecycling
        = {"enable_slice_recycling",
           "false",
           "Reuse garbage-collected and race-discarded slices from a per-store pool instead of destroying and reallocating them."};
    UIntOption slicePoolCapacity
        = {"slice_pool_capacity",
           "0",
           "Upper bound on the number of retired slices one store keeps for reuse. 0 means the steady-state number of live slices, "
           "windowSize / windowSlide."};

    BoolOption enableSliceGroupCreation
        = {"enable_slice_group_creation",
           "false",
           "Elect one worker thread to create groups of slices ahead of the stream while other threads defer their buffers. Forces the "
           "watermark assigner into its own pipeline so the buffer's min/max event timestamps reach the window build."};
    UIntOption maxSliceGroupSize
        = {"max_slice_group_size",
           "64",
           "Upper bound on the number of slices one winner creates ahead of the stream in a single group.",
           {std::make_shared<NonZeroValidation>()}};

private:
    std::vector<BaseOption*> getOptions() override
    {
        return {&enableSliceRecycling, &slicePoolCapacity, &enableSliceGroupCreation, &maxSliceGroupSize};
    }
};
}
