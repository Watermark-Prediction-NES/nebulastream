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

#include <compare>
#include <cstdint>
#include <functional>
#include <Identifiers/Identifiers.hpp>
#include <Time/Timestamp.hpp>

namespace NES
{

/// Discriminator for which slot inside a slice a spilled object refers to.
/// `index` (in SpillObjectKey) disambiguates within a role, e.g. variable-column ordinal.
enum class SpillRole : uint16_t
{
    NljLeft,
    NljRight,
    AggregateState,
    HashJoinKeys,
    HashJoinPayload,
    VarColumn,
};

/// Canonical key for a single on-storage object owned by the spill subsystem.
/// POD; hashable; no heap allocation.
struct SpillObjectKey
{
    uint64_t queryId{0};
    OriginId originId{INVALID_ORIGIN_ID};
    Timestamp sliceEnd{Timestamp::INVALID_VALUE};
    WorkerThreadId thread{WorkerThreadId(WorkerThreadId::INVALID)};
    SpillRole role{SpillRole::NljLeft};
    uint16_t index{0};

    friend constexpr std::strong_ordering operator<=>(const SpillObjectKey&, const SpillObjectKey&) = default;
    friend constexpr bool operator==(const SpillObjectKey&, const SpillObjectKey&) = default;
};

}

namespace std
{
template <>
struct hash<NES::SpillObjectKey>
{
    std::size_t operator()(const NES::SpillObjectKey& key) const noexcept
    {
        const std::size_t a = std::hash<uint64_t>{}(key.queryId);
        const std::size_t b = std::hash<uint64_t>{}(key.originId.getRawValue());
        const std::size_t c = std::hash<uint64_t>{}(key.sliceEnd.getRawValue());
        const std::size_t d = std::hash<uint32_t>{}(key.thread.getRawValue());
        const std::size_t e = (static_cast<std::size_t>(key.role) << 16U) | key.index;
        std::size_t hash = a;
        hash ^= b + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= c + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= d + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= e + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        return hash;
    }
};
}
