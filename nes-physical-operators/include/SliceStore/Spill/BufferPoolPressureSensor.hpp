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

/// Samples pool occupancy as `1.0 - available / pooled`, i.e. 0.0 with an untouched pool and 1.0 once
/// every pooled buffer is handed out. Holds a reference to an AbstractBufferProvider owned by the
/// operator handler; the sensor's lifetime must not outlive the provider.
class BufferPoolPressureSensor final : public MemoryPressureSensor
{
public:
    explicit BufferPoolPressureSensor(AbstractBufferProvider& provider) noexcept : provider(provider) { }

    [[nodiscard]] double sample() const noexcept override
    {
        const auto pooled = provider.getNumOfPooledBuffers();
        if (pooled == 0)
        {
            return 0.0;
        }
        const auto available = static_cast<double>(provider.getNumberOfAvailableBuffers());
        const auto pressure = 1.0 - (available / static_cast<double>(pooled));
        return pressure < 0.0 ? 0.0 : (pressure > 1.0 ? 1.0 : pressure);
    }

private:
    AbstractBufferProvider& provider;
};

}
