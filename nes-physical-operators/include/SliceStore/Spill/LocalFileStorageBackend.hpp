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

#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>

namespace NES
{

/// File-per-object storage backend. Each SpillObjectKey maps to one file under `spillDirectory` named
/// nes_spill_q{queryId}_o{originId}_s{sliceEnd}_t{thread}_r{roleIndex}_i{index}.dat. Writes go through
/// a std::ofstream; reads through a std::ifstream. All I/O is synchronous; the std::future interface
/// is satisfied via pre-resolved promises. waitForCompletion() is a no-op (there is no backend-side queue —
/// caller-side futures are the synchronisation point). This is a deliberate simplification vs. the
/// thesis branch's Boost.Asio coroutines, which were the source of B1/B4/B6.
class LocalFileStorageBackend final : public StorageBackend
{
public:
    explicit LocalFileStorageBackend(std::filesystem::path spillDirectory);

    [[nodiscard]] std::unique_ptr<SpillWriter> openWrite(const SpillObjectKey& key) override;
    [[nodiscard]] std::expected<std::unique_ptr<SpillReader>, IoError> openRead(const SpillObjectKey& key) override;
    [[nodiscard]] std::future<std::expected<void, IoError>> removeAsync(const SpillObjectKey& key) override;
    void waitForCompletion(std::optional<SpillObjectKey> key) override;

    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return spillDirectory; }

    /// Static helper: build the file path for a key. Public so tests can verify.
    [[nodiscard]] static std::filesystem::path pathFor(const std::filesystem::path& base, const SpillObjectKey& key);

private:
    std::filesystem::path spillDirectory;
};

/// Force-link marker. Static-library linkers drop TUs whose only symbols are anonymous-namespace
/// globals (the StorageBackendRegistrar instance). Call this from any code path that wants to ensure
/// the "local-file" backend registers itself at static init.
int forceLinkLocalFileStorageBackend();

}
