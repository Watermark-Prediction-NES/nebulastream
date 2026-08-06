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

#include <StateReduction/CompressionAlgorithm.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include <zstd.h>
#include <ErrorHandling.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

namespace
{

/// Hands the bytes back unchanged. Not a placeholder: it is the control for every compression
/// measurement, and it is what makes the Spill leaf (store verbatim) share one code path with
/// CompressAndSpill instead of needing its own.
class NoCompressionAlgorithm final : public CompressionAlgorithm
{
public:
    [[nodiscard]] std::vector<std::byte> compress(const std::span<const std::byte> raw) const override { return {raw.begin(), raw.end()}; }

    [[nodiscard]] std::vector<std::byte> decompress(const std::span<const std::byte> compressed, const uint64_t rawSize) const override
    {
        INVARIANT(compressed.size() == rawSize, "Uncompressed payload of {} bytes should have been {}", compressed.size(), rawSize);
        return {compressed.begin(), compressed.end()};
    }
};

class ZstdCompressionAlgorithm final : public CompressionAlgorithm
{
public:
    explicit ZstdCompressionAlgorithm(const int level) : level(level) { }

    [[nodiscard]] std::vector<std::byte> compress(const std::span<const std::byte> raw) const override
    {
        std::vector<std::byte> compressed(ZSTD_compressBound(raw.size()));
        const auto written = ZSTD_compress(compressed.data(), compressed.size(), raw.data(), raw.size(), level);
        if (ZSTD_isError(written) != 0U)
        {
            throw CannotSerialize("zstd failed to compress {} bytes: {}", raw.size(), ZSTD_getErrorName(written));
        }
        compressed.resize(written);
        return compressed;
    }

    [[nodiscard]] std::vector<std::byte> decompress(const std::span<const std::byte> compressed, const uint64_t rawSize) const override
    {
        std::vector<std::byte> raw(rawSize);
        const auto written = ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size());
        if (ZSTD_isError(written) != 0U)
        {
            throw CannotDeserialize("zstd failed to decompress {} bytes: {}", compressed.size(), ZSTD_getErrorName(written));
        }
        if (written != rawSize)
        {
            throw CannotDeserialize("zstd produced {} bytes, expected {}", written, rawSize);
        }
        return raw;
    }

private:
    int level;
};

}

std::unique_ptr<CompressionAlgorithm> CompressionAlgorithm::create(const CompressionAlgorithmType type, const int level)
{
    switch (type)
    {
        case CompressionAlgorithmType::NONE:
            return std::make_unique<NoCompressionAlgorithm>();
        case CompressionAlgorithmType::ZSTD:
            return std::make_unique<ZstdCompressionAlgorithm>(level);
    }
    std::unreachable();
}

}
