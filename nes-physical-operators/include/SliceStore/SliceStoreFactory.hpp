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
#include <memory>
#include <SliceStore/Spill/SpillConfiguration.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>
#include <SliceCacheConfiguration.hpp>

namespace NES
{

class AbstractBufferProvider;

/// Single entry point for constructing slice stores. When spill is disabled (default), returns a
/// plain DefaultTimeBasedSliceStore. When enabled, returns a SpillingTimeBasedSliceStore decorated
/// over a DefaultTimeBasedSliceStore with policy/backend/sensor resolved from their registries.
///
/// The factory is the ONLY place the lowering rules need to call — they pass through the
/// SpillConfiguration verbatim, no other changes are required at call sites.
class SliceStoreFactory
{
public:
    /// `bufferProvider` is required only when spill is enabled (sensor needs it). Pass nullptr when
    /// disabled; nullptr with enabled spill is a precondition violation.
    [[nodiscard]] static std::unique_ptr<WindowSlicesStoreInterface> create(
        const SpillConfiguration& spillConfig,
        uint64_t windowSize,
        uint64_t windowSlide,
        SliceCacheConfiguration sliceCacheConfig,
        AbstractBufferProvider* bufferProvider = nullptr);
};

}
