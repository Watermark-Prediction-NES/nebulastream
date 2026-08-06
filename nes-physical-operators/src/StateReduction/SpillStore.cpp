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

#include <StateReduction/SpillStore.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <span>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include <folly/Synchronized.h>
#include <ErrorHandling.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{

namespace
{

std::filesystem::path fileFor(const std::filesystem::path& directory, const uint64_t sliceEnd)
{
    return directory / fmt::format("slice_{}.state", sliceEnd);
}

/// Keeps reduced state on the C++ heap.
///
/// Worth having even though it frees no address space: the TupleBuffers a slice was built from do go
/// back to the buffer provider, and a compressed copy on the heap is smaller than what it replaced. It is
/// also the only store that works without a writable disk, and it keeps compression measurements
/// independent of the I/O path.
class InMemorySpillStore final : public SpillStore
{
public:
    void put(const uint64_t sliceEnd, const std::span<const std::byte> bytes) override
    {
        auto locked = objects.wlock();
        auto& slot = (*locked)[sliceEnd];
        totalBytes += bytes.size();
        totalBytes -= slot.size();
        slot.assign(bytes.begin(), bytes.end());
    }

    [[nodiscard]] std::vector<std::byte> get(const uint64_t sliceEnd) override
    {
        auto locked = objects.rlock();
        const auto it = locked->find(sliceEnd);
        if (it == locked->end())
        {
            throw CannotDeserialize("no reduced state stored for slice ending at {}", sliceEnd);
        }
        return it->second;
    }

    void erase(const uint64_t sliceEnd) override
    {
        auto locked = objects.wlock();
        if (const auto it = locked->find(sliceEnd); it != locked->end())
        {
            totalBytes -= it->second.size();
            locked->erase(it);
        }
    }

    [[nodiscard]] uint64_t storedBytes() const override { return totalBytes.load(std::memory_order_relaxed); }

private:
    folly::Synchronized<std::unordered_map<uint64_t, std::vector<std::byte>>> objects;
    /// Atomic rather than a plain member under `objects`' lock, so storedBytes() can stay const and
    /// lock-free — the memory budget reads it on every watermark advance.
    std::atomic<uint64_t> totalBytes{0};
};

/// Writes reduced state to one file per slice, in a directory it owns outright.
///
/// One file per slice rather than one appended log, because slices are erased out of order as their
/// windows close: an append-only file would turn every erase into compaction work for no benefit, and a
/// restore into a seek through unrelated state.
class LocalFileSpillStore final : public SpillStore
{
public:
    explicit LocalFileSpillStore(std::filesystem::path directory) : directory(std::move(directory))
    {
        std::error_code errorCode;
        /// remove_all first: a directory left behind by a worker that was killed would otherwise hand
        /// this store files it never wrote, under slice ends it may later write itself.
        std::ignore = std::filesystem::remove_all(this->directory, errorCode);
        std::filesystem::create_directories(this->directory, errorCode);
        if (errorCode)
        {
            throw CannotSerialize("cannot create spill directory {}: {}", this->directory.string(), errorCode.message());
        }
    }

    ~LocalFileSpillStore() override
    {
        std::error_code errorCode;
        std::ignore = std::filesystem::remove_all(directory, errorCode);
    }

    LocalFileSpillStore(const LocalFileSpillStore&) = delete;
    LocalFileSpillStore(LocalFileSpillStore&&) = delete;
    LocalFileSpillStore& operator=(const LocalFileSpillStore&) = delete;
    LocalFileSpillStore& operator=(LocalFileSpillStore&&) = delete;

    void put(const uint64_t sliceEnd, const std::span<const std::byte> bytes) override
    {
        const auto path = fileFor(directory, sliceEnd);
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        if (not out)
        {
            throw CannotSerialize("cannot open {} for writing reduced state", path.string());
        }
        /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        out.close();
        if (not out)
        {
            throw CannotSerialize("failed writing {} bytes of reduced state to {}", bytes.size(), path.string());
        }

        const auto locked = sizes.wlock();
        auto& previous = (*locked)[sliceEnd];
        totalBytes += bytes.size();
        totalBytes -= previous;
        previous = bytes.size();
    }

    [[nodiscard]] std::vector<std::byte> get(const uint64_t sliceEnd) override
    {
        const auto path = fileFor(directory, sliceEnd);
        std::ifstream in{path, std::ios::binary | std::ios::ate};
        if (not in)
        {
            throw CannotDeserialize("no reduced state stored at {}", path.string());
        }
        const auto size = static_cast<std::streamsize>(in.tellg());
        in.seekg(0);

        std::vector<std::byte> bytes(static_cast<size_t>(size));
        /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        if (size > 0 && not in.read(reinterpret_cast<char*>(bytes.data()), size))
        {
            throw CannotDeserialize("failed reading {} bytes of reduced state from {}", size, path.string());
        }
        return bytes;
    }

    void erase(const uint64_t sliceEnd) override
    {
        std::error_code errorCode;
        std::ignore = std::filesystem::remove(fileFor(directory, sliceEnd), errorCode);

        auto locked = sizes.wlock();
        if (const auto it = locked->find(sliceEnd); it != locked->end())
        {
            totalBytes -= it->second;
            locked->erase(it);
        }
    }

    [[nodiscard]] uint64_t storedBytes() const override { return totalBytes.load(std::memory_order_relaxed); }

private:
    std::filesystem::path directory;
    folly::Synchronized<std::unordered_map<uint64_t, uint64_t>> sizes;
    std::atomic<uint64_t> totalBytes{0};
};

}

std::unique_ptr<SpillStore> SpillStore::create(const SpillStoreType type, const std::filesystem::path& directory)
{
    switch (type)
    {
        case SpillStoreType::IN_MEMORY:
            return std::make_unique<InMemorySpillStore>();
        case SpillStoreType::LOCAL_FILE:
            return std::make_unique<LocalFileSpillStore>(directory);
    }
    std::unreachable();
}

}
