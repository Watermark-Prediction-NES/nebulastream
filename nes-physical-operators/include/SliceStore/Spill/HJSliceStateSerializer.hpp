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
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>

namespace NES
{

/// Spills HJSlice state by iterating each per-(slot, side) ChainedHashMap, walking every chain,
/// and streaming a (hash, key, value) record per entry through the StorageBackend. One file per
/// non-empty hashmap. Restore lazy-creates the hashmap via getHashMapPtrOrCreate and re-inserts
/// each entry via ChainedHashMap::insertEntry — pointer relocation is eliminated by construction.
///
/// **Scope:** fixed-size keys and fixed-size values only. If a hashmap has any populated
/// varSizedSpace, spill returns IoErrorCode::TransientIo with a "var-sized HJ spill not yet
/// implemented" message. Variable-sized support is a follow-up PR (requires per-field schema
/// metadata that NLJ also did not need).
///
/// **Threading:** ChainedHashMap is not thread-safe. Spill / restore assume the slice is
/// quiescent — no build pipeline is concurrently inserting into the hashmap.
///
/// **Endianness:** native; files are not portable across architectures.
class HJSliceStateSerializer final : public SliceStateSerializer
{
public:
    HJSliceStateSerializer() = default;

    [[nodiscard]] std::future<std::expected<SpilledSliceHandle, IoError>>
    spill(Slice& slice, StorageBackend& backend, AbstractBufferProvider& buffers) override;

    [[nodiscard]] std::future<std::expected<void, IoError>>
    restore(Slice& slice, const SpilledSliceHandle& handle, StorageBackend& backend, AbstractBufferProvider& buffers) override;

    [[nodiscard]] uint64_t residentBytes(const Slice& slice) const noexcept override;
};

/// Force-link marker. Static-library linkers drop TUs whose only symbols are anonymous-namespace
/// globals (our registrar instance). Call this from any code path that wants to ensure the
/// HJSliceStateSerializer registers itself at static init.
int forceLinkHJSerializer();

}
