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

#include <Runtime/BufferTreeCodec.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <ErrorHandling.hpp>

namespace NES::BufferTreeCodec
{

namespace
{

template <typename T>
requires(std::is_trivially_copyable_v<T>)
void append(std::vector<std::byte>& out, const T value)
{
    const auto* const bytes = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
requires(std::is_trivially_copyable_v<T>)
T take(std::span<const std::byte>& in)
{
    if (in.size() < sizeof(T))
    {
        throw CannotDeserialize("buffer tree stream truncated: wanted {} header bytes, {} left", sizeof(T), in.size());
    }
    T value;
    std::memcpy(&value, in.data(), sizeof(T));
    in = in.subspan(sizeof(T));
    return value;
}

}

void write(std::vector<std::byte>& out, const TupleBuffer& buffer)
{
    const auto payload = buffer.getAvailableMemoryArea();
    const auto numberOfChildren = buffer.getNumberOfChildBuffers();

    append<uint32_t>(out, static_cast<uint32_t>(payload.size()));
    append<uint64_t>(out, buffer.getNumberOfTuples());
    append<uint32_t>(out, numberOfChildren);
    out.insert(out.end(), payload.begin(), payload.end());

    /// Depth-first in index order. read() appends children in the same order, and storeChildBuffer
    /// assigns indices by insertion, so every ChildBufferIndex in the payload stays valid.
    for (uint32_t childIdx = 0; childIdx < numberOfChildren; ++childIdx)
    {
        write(out, buffer.loadChildBuffer(ChildBufferIndex{childIdx}));
    }
}

TupleBuffer read(std::span<const std::byte>& in, AbstractBufferProvider& bufferProvider)
{
    const auto size = take<uint32_t>(in);
    const auto numberOfTuples = take<uint64_t>(in);
    const auto numberOfChildren = take<uint32_t>(in);

    if (in.size() < size)
    {
        throw CannotDeserialize("buffer tree stream truncated: wanted {} payload bytes, {} left", size, in.size());
    }

    auto restored = bufferProvider.getUnpooledBuffer(size);
    if (not restored.has_value())
    {
        throw BufferAllocationFailure("no unpooled TupleBuffer of {} bytes available to restore state", size);
    }

    std::memcpy(restored->getAvailableMemoryArea().data(), in.data(), size);
    in = in.subspan(size);
    restored->setNumberOfTuples(numberOfTuples);

    for (uint32_t childIdx = 0; childIdx < numberOfChildren; ++childIdx)
    {
        auto child = read(in, bufferProvider);
        const auto assignedIdx = restored->storeChildBuffer(child);
        INVARIANT(
            assignedIdx == ChildBufferIndex{childIdx},
            "restored child landed at index {} instead of {}, every ChildBufferIndex in the payload is now wrong",
            assignedIdx,
            childIdx);
    }

    return std::move(restored.value());
}

uint64_t encodedSize(const TupleBuffer& buffer)
{
    constexpr uint64_t headerSize = sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t);
    const auto numberOfChildren = buffer.getNumberOfChildBuffers();

    uint64_t total = headerSize + buffer.getAvailableMemoryArea().size();
    for (uint32_t childIdx = 0; childIdx < numberOfChildren; ++childIdx)
    {
        total += encodedSize(buffer.loadChildBuffer(ChildBufferIndex{childIdx}));
    }
    return total;
}

}
