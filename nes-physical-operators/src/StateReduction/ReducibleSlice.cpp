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

#include <StateReduction/ReducibleSlice.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/BufferTreeCodec.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <ErrorHandling.hpp>

namespace NES::SliceStateCodec
{

namespace
{
constexpr std::byte SlotAbsent{0};
constexpr std::byte SlotPresent{1};

template <typename T>
requires(std::is_trivially_copyable_v<T>)
void append(std::vector<std::byte>& out, const T value)
{
    const auto* const bytes = reinterpret_cast<const std::byte*>(&value); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
requires(std::is_trivially_copyable_v<T>)
T take(std::span<const std::byte>& in)
{
    if (in.size() < sizeof(T))
    {
        throw CannotDeserialize("slice state stream truncated: wanted {} header bytes, {} left", sizeof(T), in.size());
    }
    T value;
    std::memcpy(&value, in.data(), sizeof(T));
    in = in.subspan(sizeof(T));
    return value;
}
}

void encodeAndRelease(std::vector<std::byte>& out, std::vector<TupleBuffer>& slots)
{
    append<uint32_t>(out, static_cast<uint32_t>(slots.size()));
    for (auto& slot : slots)
    {
        /// A slot can legitimately be empty: hash maps are allocated on first touch, so a worker thread
        /// that never saw a record for this slice leaves its slot untouched.
        if (not slot)
        {
            out.push_back(SlotAbsent);
            continue;
        }
        out.push_back(SlotPresent);
        BufferTreeCodec::write(out, slot);
        /// Assigning an empty buffer drops the last reference to this tree and hands every page in it
        /// back to the buffer provider. This is the entire point of the exercise.
        slot = TupleBuffer{};
    }
}

void decodeInPlace(std::span<const std::byte>& in, std::vector<TupleBuffer>& slots, AbstractBufferProvider& bufferProvider)
{
    const auto encodedSlotCount = take<uint32_t>(in);
    INVARIANT(
        encodedSlotCount == slots.size(),
        "Slice was encoded with {} slots but now has {}; outstanding pointers into the slot vector make resizing it unsafe",
        encodedSlotCount,
        slots.size());

    for (auto& slot : slots)
    {
        const auto presence = take<std::byte>(in);
        if (presence == SlotAbsent)
        {
            continue;
        }
        slot = BufferTreeCodec::read(in, bufferProvider);
    }
}

uint64_t residentBytes(const std::vector<TupleBuffer>& slots)
{
    uint64_t total = 0;
    for (const auto& slot : slots)
    {
        if (not slot)
        {
            continue;
        }
        /// encodedSize walks the same tree a reduction would write, so this counts exactly the bytes a
        /// reduction would free — including variable-sized child buffers, which a page-count estimate
        /// would miss entirely on varsized-heavy workloads. It costs one walk per slice per watermark
        /// advance, not per record.
        total += BufferTreeCodec::encodedSize(slot);
    }
    return total;
}

}
