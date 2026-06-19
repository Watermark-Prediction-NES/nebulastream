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

/// Spills NLJSlice state by iterating each per-worker-thread PagedVector on both build sides and
/// streaming each page's bytes (preceded by a small fixed header) to its own SpillObjectKey.
/// One file per (slice, thread, side, pageIndex) — restore is embarrassingly parallel.
///
/// On-disk format per object:
///   header: numberOfTuples (uint64), bufferSize (uint64)
///   payload: `bufferSize` bytes of the TupleBuffer's data region
///
/// QueryId in the SpillObjectKey is supplied by the decorator (it knows the host query).
/// For v1 we use a fixed dummy queryId of 0 — the decorator overrides via SpillContext when wiring.
class NLJSliceStateSerializer final : public SliceStateSerializer
{
public:
    NLJSliceStateSerializer() = default;

    [[nodiscard]] std::future<std::expected<SpilledSliceHandle, IoError>>
    spill(Slice& slice, StorageBackend& backend, AbstractBufferProvider& buffers) override;

    [[nodiscard]] std::future<std::expected<void, IoError>>
    restore(Slice& slice, const SpilledSliceHandle& handle, StorageBackend& backend, AbstractBufferProvider& buffers) override;

    [[nodiscard]] uint64_t residentBytes(const Slice& slice) const noexcept override;
};

/// Force-link marker. Static-library linkers drop TUs whose only symbols are anonymous-namespace
/// globals (our registrar instance). Call this from any code path that wants to ensure the
/// NLJSliceStateSerializer registers itself at static init.
int forceLinkNLJSerializer();

}
