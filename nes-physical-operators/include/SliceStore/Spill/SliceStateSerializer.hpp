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
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/SpillObjectKey.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>

namespace NES
{

/// Opaque handle returned from a successful spill. The decorator stashes one per spilled slice and
/// presents it back to the serializer's restore() method later. `stateFingerprint` guards against
/// bug-induced key collisions (e.g. two queries reusing the same SpillObjectKey range).
struct SpilledSliceHandle
{
    std::vector<SpillObjectKey> keys;
    uint64_t totalBytes{0};
    uint64_t stateFingerprint{0};
};

/// Per-Slice-subclass (de)serialization. One implementation per concrete Slice subclass that wants
/// spill support; registered against std::type_index{typeid(SubSlice)} in SliceStateSerializerRegistry.
/// The store never dynamic_pointer_casts — dispatch goes through the registry. Fixes B9.
class SliceStateSerializer
{
public:
    virtual ~SliceStateSerializer() = default;

    SliceStateSerializer() = default;
    SliceStateSerializer(const SliceStateSerializer&) = delete;
    SliceStateSerializer(SliceStateSerializer&&) = delete;
    SliceStateSerializer& operator=(const SliceStateSerializer&) = delete;
    SliceStateSerializer& operator=(SliceStateSerializer&&) = delete;

    /// Streams `slice`'s pages through `backend`. On success, the serializer releases the in-memory
    /// pages of the slice and returns a handle. On failure, the slice's resident pages MUST remain
    /// intact — the caller has not lost data.
    [[nodiscard]] virtual std::future<std::expected<SpilledSliceHandle, IoError>>
    spill(Slice& slice, StorageBackend& backend, AbstractBufferProvider& buffers) = 0;

    [[nodiscard]] virtual std::future<std::expected<void, IoError>>
    restore(Slice& slice, const SpilledSliceHandle& handle, StorageBackend& backend, AbstractBufferProvider& buffers) = 0;

    [[nodiscard]] virtual uint64_t residentBytes(const Slice& slice) const noexcept = 0;
};

}
