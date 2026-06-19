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
#include <expected>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <SliceStore/Spill/SpillObjectKey.hpp>

namespace NES
{

enum class IoErrorCode : uint8_t
{
    TransientIo,
    FsFull,
    NotFound,
    Corrupted,
    ShuttingDown,
};

struct IoError
{
    IoErrorCode code{IoErrorCode::TransientIo};
    std::string message{};
};

/// Streaming writer for a single spill object. Append-only; pages are flushed page-by-page so the
/// serializer never has to materialise the whole slice in RAM (the failure mode of the thesis branch).
/// Stable bytes are NOT guaranteed visible to a reader until close() resolves.
class SpillWriter
{
public:
    virtual ~SpillWriter() = default;

    SpillWriter() = default;
    SpillWriter(const SpillWriter&) = delete;
    SpillWriter(SpillWriter&&) = delete;
    SpillWriter& operator=(const SpillWriter&) = delete;
    SpillWriter& operator=(SpillWriter&&) = delete;

    [[nodiscard]] virtual std::future<std::expected<void, IoError>> append(std::span<const std::byte> bytes) = 0;

    /// Resolves when the object is durable enough that a subsequent openRead sees its bytes.
    [[nodiscard]] virtual std::future<std::expected<void, IoError>> close() = 0;
};

/// Streaming reader for a single spill object. readNext returns the count actually filled (0 == EOF).
class SpillReader
{
public:
    virtual ~SpillReader() = default;

    SpillReader() = default;
    SpillReader(const SpillReader&) = delete;
    SpillReader(SpillReader&&) = delete;
    SpillReader& operator=(const SpillReader&) = delete;
    SpillReader& operator=(SpillReader&&) = delete;

    [[nodiscard]] virtual std::future<std::expected<uint64_t, IoError>> readNext(std::span<std::byte> dst) = 0;
};

/// Backend abstraction for spilled-state I/O. The slice store talks to a StorageBackend, not directly to files.
/// All failure cases are surfaced via std::expected — silent loss is not possible by construction (fixes B1).
class StorageBackend
{
public:
    virtual ~StorageBackend() = default;

    StorageBackend() = default;
    StorageBackend(const StorageBackend&) = delete;
    StorageBackend(StorageBackend&&) = delete;
    StorageBackend& operator=(const StorageBackend&) = delete;
    StorageBackend& operator=(StorageBackend&&) = delete;

    [[nodiscard]] virtual std::unique_ptr<SpillWriter> openWrite(const SpillObjectKey& key) = 0;

    [[nodiscard]] virtual std::expected<std::unique_ptr<SpillReader>, IoError> openRead(const SpillObjectKey& key) = 0;

    [[nodiscard]] virtual std::future<std::expected<void, IoError>> removeAsync(const SpillObjectKey& key) = 0;

    /// Block until every in-flight op completes (or only ops on `key`, if provided). Called from dtors and deleteState.
    virtual void waitForCompletion(std::optional<SpillObjectKey> key) = 0;
};

}
