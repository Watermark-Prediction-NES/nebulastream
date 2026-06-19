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

#include <SliceStore/Spill/LocalFileStorageBackend.hpp>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>
#include <SliceStore/Spill/StorageBackendRegistry.hpp>
#include <Util/Logger/Logger.hpp>

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

std::filesystem::path LocalFileStorageBackend::pathFor(const std::filesystem::path& base, const SpillObjectKey& key)
{
    return base
        / std::format(
               "nes_spill_q{}_o{}_s{}_t{}_r{}_i{}.dat",
               key.queryId,
               key.originId.getRawValue(),
               key.sliceEnd.getRawValue(),
               key.thread.getRawValue(),
               static_cast<uint32_t>(key.role),
               key.index);
}

LocalFileStorageBackend::LocalFileStorageBackend(std::filesystem::path spillDirectory_) : spillDirectory(std::move(spillDirectory_))
{
    std::error_code ec;
    std::filesystem::create_directories(spillDirectory, ec);
    if (ec)
    {
        NES_WARNING("LocalFileStorageBackend: failed to create spill dir {}: {}", spillDirectory.string(), ec.message());
    }
}

/// Streaming writer over std::ofstream. append() writes synchronously and returns a resolved future.
/// close() flushes and closes the stream.
class LocalFileSpillWriter final : public SpillWriter
{
public:
    LocalFileSpillWriter(std::filesystem::path path)
        : path(std::move(path)), stream(this->path, std::ios::out | std::ios::binary | std::ios::trunc)
    {
        if (!stream.is_open())
        {
            openFailed = true;
            errorMessage = std::strerror(errno);
        }
    }

    [[nodiscard]] std::future<std::expected<void, IoError>> append(std::span<const std::byte> bytes) override
    {
        if (openFailed)
        {
            return ready<std::expected<void, IoError>>(std::unexpected{IoError{IoErrorCode::TransientIo, errorMessage}});
        }
        if (closed)
        {
            return ready<std::expected<void, IoError>>(std::unexpected{IoError{IoErrorCode::TransientIo, "Writer already closed"}});
        }
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size_bytes()));
        if (!stream.good())
        {
            return ready<std::expected<void, IoError>>(std::unexpected{IoError{IoErrorCode::TransientIo, "write() failed"}});
        }
        return ready<std::expected<void, IoError>>({});
    }

    [[nodiscard]] std::future<std::expected<void, IoError>> close() override
    {
        if (closed || openFailed)
        {
            return ready<std::expected<void, IoError>>({});
        }
        closed = true;
        stream.flush();
        stream.close();
        if (stream.bad())
        {
            return ready<std::expected<void, IoError>>(std::unexpected{IoError{IoErrorCode::TransientIo, "close() failed"}});
        }
        return ready<std::expected<void, IoError>>({});
    }

private:
    std::filesystem::path path;
    std::ofstream stream;
    bool openFailed{false};
    bool closed{false};
    std::string errorMessage;
};

/// Streaming reader over std::ifstream.
class LocalFileSpillReader final : public SpillReader
{
public:
    LocalFileSpillReader(std::ifstream&& stream) : stream(std::move(stream)) { }

    [[nodiscard]] std::future<std::expected<uint64_t, IoError>> readNext(std::span<std::byte> dst) override
    {
        if (dst.empty())
        {
            return ready<std::expected<uint64_t, IoError>>(uint64_t{0});
        }
        stream.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(dst.size_bytes()));
        const auto bytesRead = static_cast<uint64_t>(stream.gcount());
        if (stream.bad())
        {
            return ready<std::expected<uint64_t, IoError>>(std::unexpected{IoError{IoErrorCode::TransientIo, "read() failed"}});
        }
        return ready<std::expected<uint64_t, IoError>>(bytesRead);
    }

private:
    std::ifstream stream;
};

std::unique_ptr<SpillWriter> LocalFileStorageBackend::openWrite(const SpillObjectKey& key)
{
    return std::make_unique<LocalFileSpillWriter>(pathFor(spillDirectory, key));
}

std::expected<std::unique_ptr<SpillReader>, IoError> LocalFileStorageBackend::openRead(const SpillObjectKey& key)
{
    const auto path = pathFor(spillDirectory, key);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
    {
        return std::unexpected{IoError{IoErrorCode::NotFound, std::format("file not found: {}", path.string())}};
    }
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    if (!stream.is_open())
    {
        return std::unexpected{IoError{IoErrorCode::TransientIo, std::strerror(errno)}};
    }
    return std::make_unique<LocalFileSpillReader>(std::move(stream));
}

std::future<std::expected<void, IoError>> LocalFileStorageBackend::removeAsync(const SpillObjectKey& key)
{
    const auto path = pathFor(spillDirectory, key);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
    {
        return ready<std::expected<void, IoError>>(std::unexpected{IoError{IoErrorCode::TransientIo, ec.message()}});
    }
    return ready<std::expected<void, IoError>>({});
}

void LocalFileStorageBackend::waitForCompletion(std::optional<SpillObjectKey> /*key*/)
{
    /// All operations are synchronous; nothing to wait on.
}

namespace
{
const StorageBackendRegistrar registrar{
    "local-file",
    [](const StorageBackendArgs& args) -> std::shared_ptr<StorageBackend>
    { return std::make_shared<LocalFileStorageBackend>(args.spillDirectory); }};
}

int forceLinkLocalFileStorageBackend()
{
    return 1;
}

}
