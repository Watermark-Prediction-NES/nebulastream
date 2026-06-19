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

#include <SliceStore/Spill/InMemoryStorageBackend.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/Spill/StorageBackendRegistry.hpp>

namespace NES
{

namespace
{
template <typename T>
std::future<T> ready(T value)
{
    std::promise<T> promise;
    auto future = promise.get_future();
    promise.set_value(std::move(value));
    return future;
}
}

/// Streaming writer that accumulates bytes into a vector and on close() commits them to the backend's map.
/// Append is in-memory only; the bytes become visible to readers only after close() resolves.
class InMemorySpillWriter final : public SpillWriter
{
public:
    InMemorySpillWriter(InMemoryStorageBackend& backend, SpillObjectKey key) : backend(backend), key(std::move(key)) { }

    [[nodiscard]] std::future<std::expected<void, IoError>> append(std::span<const std::byte> bytes) override
    {
        if (closed)
        {
            return ready<std::expected<void, IoError>>(std::unexpected{IoError{IoErrorCode::TransientIo, "Writer already closed"}});
        }
        const auto prevSize = pending.size();
        pending.resize(prevSize + bytes.size_bytes());
        std::memcpy(pending.data() + prevSize, bytes.data(), bytes.size_bytes());
        return ready<std::expected<void, IoError>>({});
    }

    [[nodiscard]] std::future<std::expected<void, IoError>> close() override
    {
        if (closed)
        {
            return ready<std::expected<void, IoError>>({});
        }
        closed = true;
        backend.storage.withWLock([&](auto& map) { map[key] = std::move(pending); });
        return ready<std::expected<void, IoError>>({});
    }

private:
    InMemoryStorageBackend& backend;
    SpillObjectKey key;
    std::vector<std::byte> pending;
    bool closed{false};
};

/// Streaming reader that reads sequentially from a snapshot of the object's bytes at open time.
class InMemorySpillReader final : public SpillReader
{
public:
    InMemorySpillReader(std::vector<std::byte> snapshot) : snapshot(std::move(snapshot)) { }

    [[nodiscard]] std::future<std::expected<uint64_t, IoError>> readNext(std::span<std::byte> dst) override
    {
        const auto remaining = snapshot.size() - cursor;
        const auto toCopy = std::min<std::size_t>(remaining, dst.size_bytes());
        if (toCopy > 0)
        {
            std::memcpy(dst.data(), snapshot.data() + cursor, toCopy);
            cursor += toCopy;
        }
        return ready<std::expected<uint64_t, IoError>>(static_cast<uint64_t>(toCopy));
    }

private:
    std::vector<std::byte> snapshot;
    std::size_t cursor{0};
};

std::unique_ptr<SpillWriter> InMemoryStorageBackend::openWrite(const SpillObjectKey& key)
{
    return std::make_unique<InMemorySpillWriter>(*this, key);
}

std::expected<std::unique_ptr<SpillReader>, IoError> InMemoryStorageBackend::openRead(const SpillObjectKey& key)
{
    auto snapshot = storage.withRLock(
        [&](const auto& map) -> std::optional<std::vector<std::byte>>
        {
            if (const auto it = map.find(key); it != map.end())
            {
                return it->second;
            }
            return std::nullopt;
        });
    if (!snapshot.has_value())
    {
        return std::unexpected{IoError{IoErrorCode::NotFound, "InMemoryStorageBackend: no object for key"}};
    }
    return std::make_unique<InMemorySpillReader>(std::move(*snapshot));
}

std::future<std::expected<void, IoError>> InMemoryStorageBackend::removeAsync(const SpillObjectKey& key)
{
    storage.withWLock([&](auto& map) { map.erase(key); });
    return ready<std::expected<void, IoError>>({});
}

void InMemoryStorageBackend::waitForCompletion(std::optional<SpillObjectKey> /*key*/)
{
    /// All operations are synchronous; nothing to wait on.
}

uint64_t InMemoryStorageBackend::bytesStored(const SpillObjectKey& key) const noexcept
{
    return storage.withRLock(
        [&](const auto& map) -> uint64_t
        {
            if (const auto it = map.find(key); it != map.end())
            {
                return static_cast<uint64_t>(it->second.size());
            }
            return 0;
        });
}

std::size_t InMemoryStorageBackend::numStoredObjects() const noexcept
{
    return storage.withRLock([](const auto& map) { return map.size(); });
}

namespace
{
const StorageBackendRegistrar registrar{
    "in-memory", [](const StorageBackendArgs&) -> std::shared_ptr<StorageBackend> { return std::make_shared<InMemoryStorageBackend>(); }};
}

int forceLinkInMemoryStorageBackend()
{
    return 1;
}

}
