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

namespace NES
{

/// Returns a normalised buffer-pool pressure value in [0.0, 1.0]; 1.0 means the pool is full.
/// Sampled by the slice-store decorator once per GC tick and passed to every SpillPolicy::decide call.
class MemoryPressureSensor
{
public:
    virtual ~MemoryPressureSensor() = default;

    MemoryPressureSensor() = default;
    MemoryPressureSensor(const MemoryPressureSensor&) = delete;
    MemoryPressureSensor(MemoryPressureSensor&&) = delete;
    MemoryPressureSensor& operator=(const MemoryPressureSensor&) = delete;
    MemoryPressureSensor& operator=(MemoryPressureSensor&&) = delete;

    [[nodiscard]] virtual double sample() const noexcept = 0;
};

}
