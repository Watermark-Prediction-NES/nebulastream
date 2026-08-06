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
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <StateReductionTypes.hpp>

namespace NES
{

/// Compresses a serialised slice's bytes.
///
/// It sees a flat byte stream rather than the slice's structure, which is what keeps it swappable: an
/// implementation needs to know nothing about pages, hash maps or schemas. Implementations must be
/// callable concurrently from several worker threads.
class CompressionAlgorithm
{
public:
    virtual ~CompressionAlgorithm() = default;

    CompressionAlgorithm() = default;
    CompressionAlgorithm(const CompressionAlgorithm&) = delete;
    CompressionAlgorithm(CompressionAlgorithm&&) = delete;
    CompressionAlgorithm& operator=(const CompressionAlgorithm&) = delete;
    CompressionAlgorithm& operator=(CompressionAlgorithm&&) = delete;

    [[nodiscard]] virtual std::vector<std::byte> compress(std::span<const std::byte> raw) const = 0;

    /// @param rawSize the size compress() was given. Carried alongside rather than recovered from the
    ///        payload so that an implementation is free to use a format without a length header.
    [[nodiscard]] virtual std::vector<std::byte> decompress(std::span<const std::byte> compressed, uint64_t rawSize) const = 0;

    /// @param level implementation-defined effort knob; ignored by algorithms that have none.
    [[nodiscard]] static std::unique_ptr<CompressionAlgorithm> create(CompressionAlgorithmType type, int level);
};

}
