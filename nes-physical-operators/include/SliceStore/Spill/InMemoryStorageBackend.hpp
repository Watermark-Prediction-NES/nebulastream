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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <folly/Synchronized.h>

namespace NES
{

/// Deterministic in-memory backend. Used by contract tests and by deterministic system tests
/// (so CI doesn't depend on filesystem behaviour). All futures complete synchronously inside the
/// caller's thread; no async semantics. Bytes are persisted to a shared map keyed on SpillObjectKey.
class InMemoryStorageBackend final : public StorageBackend
{
public:
    InMemoryStorageBackend() = default;

    [[nodiscard]] std::unique_ptr<SpillWriter> openWrite(const SpillObjectKey& key) override;
    [[nodiscard]] std::expected<std::unique_ptr<SpillReader>, IoError> openRead(const SpillObjectKey& key) override;
    [[nodiscard]] std::future<std::expected<void, IoError>> removeAsync(const SpillObjectKey& key) override;
    void waitForCompletion(std::optional<SpillObjectKey> key) override;

    /// Test-only accessor: returns the number of bytes currently stored under `key`, or 0 if absent.
    [[nodiscard]] uint64_t bytesStored(const SpillObjectKey& key) const noexcept;

    /// Test-only: total objects currently stored.
    [[nodiscard]] std::size_t numStoredObjects() const noexcept;

private:
    friend class InMemorySpillWriter;
    friend class InMemorySpillReader;

    folly::Synchronized<std::unordered_map<SpillObjectKey, std::vector<std::byte>>> storage;

    /// Tracks the next-failure-after-N-bytes injection for fault-injection tests.
    /// Default 0 == never fail.
    std::atomic<uint64_t> failAfterBytes{0};
};

/// Force-link marker. Static-library linkers drop TUs whose only symbols are anonymous-namespace
/// globals (the StorageBackendRegistrar instance). Call this from any code path that wants to ensure
/// the "in-memory" backend registers itself at static init.
int forceLinkInMemoryStorageBackend();

}
