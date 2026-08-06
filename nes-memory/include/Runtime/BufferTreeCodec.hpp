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
#include <span>
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>

namespace NES::BufferTreeCodec
{

/// Flattens a TupleBuffer together with its entire child-buffer tree into a contiguous byte stream, and
/// rebuilds that tree from the stream.
///
/// This copies bytes and child-buffer structure, nothing more. It is deliberately ignorant of whatever
/// data structure those bytes belong to, which is what lets one codec serve all of them.
///
/// It works because payload that refers to another buffer does so by ChildBufferIndex rather than by
/// address. read() restores children in the order write() emitted them and TupleBuffer::storeChildBuffer
/// hands out indices in insertion order, so every index in a restored tree still denotes what it did
/// before.
///
/// What it cannot reproduce is a raw pointer stored inside a payload, since restored buffers live at new
/// addresses. A structure that keeps pointers — into its own buffers or into anything else — therefore
/// comes back structurally intact but internally stale, and its owner has to reinitialise those links
/// after read() before the structure is usable again. Whether that is needed, and what it costs, is a
/// property of the structure, not of this codec; a structure that stores only indices and offsets needs
/// nothing.
///
/// Wire format per node, native endianness (spilled state never leaves the machine that wrote it):
///     [uint32 size][uint64 numberOfTuples][uint32 numberOfChildren][size payload bytes][children...]

/// Appends the encoding of `buffer` and its whole child tree to `out`.
void write(std::vector<std::byte>& out, const TupleBuffer& buffer);

/// Decodes one node and its whole child tree from the front of `in`, advancing `in` past it.
/// Restored buffers are unpooled, so any node size round-trips regardless of the pool's buffer size.
/// Throws CannotDeserialize on a truncated or malformed stream, BufferAllocationFailure if the
/// unpooled budget cannot back the tree.
TupleBuffer read(std::span<const std::byte>& in, AbstractBufferProvider& bufferProvider);

/// Number of bytes write() would append for `buffer`. Lets callers reserve once instead of growing.
[[nodiscard]] uint64_t encodedSize(const TupleBuffer& buffer);

}
