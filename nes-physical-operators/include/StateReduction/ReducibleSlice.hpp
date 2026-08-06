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
#include <shared_mutex>
#include <span>
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>

namespace NES
{

/// A slice whose state can be moved out of resident memory and brought back.
///
/// Implementations reclaim memory by dropping the TupleBuffers they hold. That works without any
/// buffer-detaching machinery because a slice's state hangs off a small number of top-level buffers, and
/// releasing one of those releases its whole child tree — pages, per-entry vectors, variable-sized
/// payloads and all. So a reduction is: encode the trees, then overwrite each slot with an empty
/// TupleBuffer.
///
/// The slot itself must stay where it is. Slices hand out `const TupleBuffer*` pointing into their own
/// vectors, and the slice cache holds on to those pointers, so the vector may not be resized or
/// reordered by a reduction — only the buffers inside it may be swapped out.
///
/// All three state operations must be called under stateMutex(): serialize/deserialize exclusively,
/// anything that reads the buffers in shared mode.
class ReducibleSlice
{
public:
    virtual ~ReducibleSlice() = default;

    ReducibleSlice() = default;
    ReducibleSlice(const ReducibleSlice&) = delete;
    ReducibleSlice(ReducibleSlice&&) = delete;
    ReducibleSlice& operator=(const ReducibleSlice&) = delete;
    ReducibleSlice& operator=(ReducibleSlice&&) = delete;

    /// Bytes this slice currently holds in TupleBuffers, counting every child buffer.
    ///
    /// Walks the whole buffer tree, so it is only safe on a slice no build thread can still be writing
    /// to -- in practice one whose end is at or before the global build watermark. The build path takes
    /// no lock, so a concurrent walk reads a child count that grows underneath it and then loads an index
    /// that is out of range. `const` and stateMutex() are both misleading here: neither makes this safe
    /// against a writer that never synchronises.
    [[nodiscard]] virtual uint64_t residentBytes() const = 0;

    /// Encodes all state into `out` and releases the buffers holding it. A no-op if already reduced.
    virtual void serializeState(std::vector<std::byte>& out) = 0;

    /// Rebuilds the state that serializeState() wrote. A no-op if not currently reduced.
    virtual void deserializeState(std::span<const std::byte> in, AbstractBufferProvider& bufferProvider) = 0;

    /// True once the state has been encoded away and not yet brought back. Maintained by the
    /// serialize/deserialize implementations, which run under an exclusive stateMutex().
    [[nodiscard]] bool isReduced() const { return reduced; }

    [[nodiscard]] std::shared_mutex& stateMutex() const { return stateLock; }

protected:
    mutable std::shared_mutex stateLock;
    bool reduced{false};
};

/// Both slice kinds hold their state the same way — a vector of top-level TupleBuffer slots, some of
/// which may not have been touched yet — so encoding one is the same job for both.
namespace SliceStateCodec
{

/// Encodes every slot and then releases the buffers, leaving the vector the same size with empty slots.
/// The size is not changed, because outstanding `const TupleBuffer*` point into it.
void encodeAndRelease(std::vector<std::byte>& out, std::vector<TupleBuffer>& slots);

/// Restores what encodeAndRelease() wrote back into `slots`, in place. The vector must still have the
/// size it had when it was encoded.
void decodeInPlace(std::span<const std::byte>& in, std::vector<TupleBuffer>& slots, AbstractBufferProvider& bufferProvider);

/// Bytes held across all present slots, counting every child buffer.
[[nodiscard]] uint64_t residentBytes(const std::vector<TupleBuffer>& slots);

}

}
