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

#include <cstddef>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Spill/MemoryPressureSensor.hpp>

namespace NES
{

/// Samples `1.0 - getNumOfPooledBuffers() / capacity`. Holds a reference to an AbstractBufferProvider
/// owned by the operator handler; the sensor's lifetime must not outlive the provider.
class BufferPoolPressureSensor final : public MemoryPressureSensor
{
public:
    BufferPoolPressureSensor(AbstractBufferProvider& provider, std::size_t capacity) noexcept
        : provider(provider), capacity(capacity == 0 ? 1 : capacity)
    {
    }

    [[nodiscard]] double sample() const noexcept override
    {
        const auto free = static_cast<double>(provider.getNumOfPooledBuffers());
        const auto pressure = 1.0 - (free / static_cast<double>(capacity));
        return pressure < 0.0 ? 0.0 : (pressure > 1.0 ? 1.0 : pressure);
    }

private:
    AbstractBufferProvider& provider;
    std::size_t capacity;
};

}
