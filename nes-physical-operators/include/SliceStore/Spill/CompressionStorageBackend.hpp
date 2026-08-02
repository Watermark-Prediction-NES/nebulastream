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
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>

namespace NES
{

/// Decorator backend that zstd-compresses the byte stream and delegates storage to an inner backend.
/// Composing it over the in-memory backend keeps slices resident but smaller; over the local-file
/// backend it compresses, then spills. Each append() is compressed one-shot into a self-describing
/// frame [uint32 compressedLen][uint32 rawLen][compressed bytes], so the reader can stream frame by
/// frame without materialising the whole object. Registered with StorageBackendRegistry as
/// "compression"; the factory builds the inner backend from StorageBackendArgs::innerBackendName.
class CompressionStorageBackend final : public StorageBackend
{
public:
    CompressionStorageBackend(std::shared_ptr<StorageBackend> inner, uint32_t level);

    [[nodiscard]] std::unique_ptr<SpillWriter> openWrite(const SpillObjectKey& key) override;
    [[nodiscard]] std::expected<std::unique_ptr<SpillReader>, IoError> openRead(const SpillObjectKey& key) override;
    [[nodiscard]] std::future<std::expected<void, IoError>> removeAsync(const SpillObjectKey& key) override;
    void waitForCompletion(std::optional<SpillObjectKey> key) override;

private:
    std::shared_ptr<StorageBackend> inner;
    uint32_t level;
};

/// Force-link marker. Static-library linkers drop TUs whose only symbols are anonymous-namespace
/// globals (the StorageBackendRegistrar instance). Call this from any code path that wants to ensure
/// the "compression" backend registers itself at static init.
int forceLinkCompressionStorageBackend();

}
