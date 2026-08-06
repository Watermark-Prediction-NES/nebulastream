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
#include <filesystem>
#include <memory>
#include <span>
#include <vector>
#include <StateReductionTypes.hpp>

namespace NES
{

/// Where reduced slice state is parked while it is not resident: a flat keyed store holding one entry per
/// reduced slice, keyed by that slice's end timestamp.
///
/// One store serves one operator instance, created and owned by its StateReductionManager. A slice end is
/// only unique within one operator, so isolation comes from that ownership: each manager hands its store
/// a directory of its own, named after both the process and the operator instance. That is simpler than
/// widening the key, and it gives cleanup a single thing to delete. Both halves of the name matter — a
/// LOCAL_FILE store clears its directory when it is constructed, so two workers sharing a spill root
/// would otherwise delete each other's state mid-run.
///
/// Deliberately synchronous. The obvious alternative is a futures-based interface, but every backend
/// worth having here completes inline and every caller needs the bytes before it can continue, so the
/// futures would only be pre-resolved promises that each caller immediately waits on.
///
/// Implementations must tolerate concurrent calls for different slices.
class SpillStore
{
public:
    virtual ~SpillStore() = default;

    SpillStore() = default;
    SpillStore(const SpillStore&) = delete;
    SpillStore(SpillStore&&) = delete;
    SpillStore& operator=(const SpillStore&) = delete;
    SpillStore& operator=(SpillStore&&) = delete;

    /// Stores `bytes` for `sliceEnd`, replacing anything already there.
    virtual void put(uint64_t sliceEnd, std::span<const std::byte> bytes) = 0;

    /// Reads back what put() stored. Throws CannotDeserialize if nothing was stored for `sliceEnd`.
    [[nodiscard]] virtual std::vector<std::byte> get(uint64_t sliceEnd) = 0;

    /// Drops a slice's entry. Erasing a slice that was never reduced is not an error: garbage collection
    /// calls this for every slice it discards without tracking which ones were spilled.
    virtual void erase(uint64_t sliceEnd) = 0;

    /// Bytes currently held. For IN_MEMORY this is memory the operator is still using, just in a
    /// compacted form, so a memory budget has to count it; for LOCAL_FILE it is disk and does not.
    [[nodiscard]] virtual uint64_t storedBytes() const = 0;

    /// @param directory must be exclusive to this store. LOCAL_FILE creates it, writes one file per
    ///        slice into it, and removes the whole thing on destruction. IN_MEMORY ignores it.
    [[nodiscard]] static std::unique_ptr<SpillStore> create(SpillStoreType type, const std::filesystem::path& directory);
};

}
