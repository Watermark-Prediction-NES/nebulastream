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
#include <Configurations/SpillConfiguration.hpp>
#include <SliceStore/WindowSlicesStoreInterface.hpp>

namespace NES
{

class AbstractBufferProvider;

/// Single entry point for decorating a slice store with spilling. When spill is disabled (default),
/// the inner store is returned unchanged. When enabled, the inner store is wrapped in a
/// SpillingTimeBasedSliceStore with policy/backend/sensor resolved from their registries.
///
/// The wrap is deferred to runtime (`WindowBasedOperatorHandler::start()`), where the buffer
/// provider first becomes available — lowering rules build only the plain inner store.
class SliceStoreFactory
{
public:
    /// Wraps an existing in-memory `WindowSlicesStoreInterface` with a `SpillingTimeBasedSliceStore`
    /// when `spillConfig.enabled`. This is the path used by `WindowBasedOperatorHandler::start()`,
    /// where the buffer provider only becomes available at runtime. `serializerName` is the serializer
    /// registered for the Slice subclass this store holds (set on the handler in the lowering); it is
    /// resolved once here. If spill is disabled, the buffer provider is null, or no serializer is
    /// registered under `serializerName`, the inner store is returned unchanged with a `NES_WARN`.
    [[nodiscard]] static std::unique_ptr<WindowSlicesStoreInterface> wrapWithSpill(
        std::unique_ptr<WindowSlicesStoreInterface> inner,
        const SpillConfiguration& spillConfig,
        AbstractBufferProvider* bufferProvider,
        const std::string& serializerName);
};

}
