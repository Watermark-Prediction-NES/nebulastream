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

#include <StateReduction/StateReductionCostModel.hpp>
#include <StateReductionTypes.hpp>

#include <cmath>
#include <cstdint>
#include <utility>

namespace NES
{

void StageCost::applyCorrection(const double correction, const double weight)
{
    if (not std::isfinite(correction) || correction <= 0.0)
    {
        return;
    }
    const double blended = 1.0 + (weight * (correction - 1.0));
    fixedNanos *= blended;
    nanosPerByte *= blended;
}

double StateReductionCostModel::reduceNanos(const StateDecision decision, const uint64_t bytes) const
{
    const auto storedBytes = static_cast<uint64_t>(static_cast<double>(bytes) * compressionRatio);
    switch (decision)
    {
        case StateDecision::KeepInMemory:
            return 0.0;
        case StateDecision::CompressInMemory:
            return compress.nanosFor(bytes);
        case StateDecision::Spill:
            return spillWrite.nanosFor(bytes);
        case StateDecision::CompressAndSpill:
            return compress.nanosFor(bytes) + spillWrite.nanosFor(storedBytes);
    }
    std::unreachable();
}

double StateReductionCostModel::restoreNanos(const StateDecision decision, const uint64_t storedBytes) const
{
    switch (decision)
    {
        case StateDecision::KeepInMemory:
            return 0.0;
        case StateDecision::CompressInMemory:
            return decompress.nanosFor(storedBytes);
        case StateDecision::Spill:
            return spillRead.nanosFor(storedBytes);
        case StateDecision::CompressAndSpill:
            return spillRead.nanosFor(storedBytes) + decompress.nanosFor(storedBytes);
    }
    std::unreachable();
}

double StateReductionCostModel::roundTripNanos(const StateDecision decision, const uint64_t bytes) const
{
    const auto compressedBytes = static_cast<uint64_t>(static_cast<double>(bytes) * compressionRatio);
    switch (decision)
    {
        case StateDecision::KeepInMemory:
            return 0.0;
        case StateDecision::CompressInMemory:
        case StateDecision::CompressAndSpill:
            return reduceNanos(decision, bytes) + restoreNanos(decision, compressedBytes);
        case StateDecision::Spill:
            /// Spilling stores the state uncompressed, so the round trip moves the full size both ways.
            return reduceNanos(decision, bytes) + restoreNanos(decision, bytes);
    }
    std::unreachable();
}

}
