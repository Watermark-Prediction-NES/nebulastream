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

/// The vocabulary shared between the state-reduction configuration and the components it selects.
///
/// It lives here, next to SliceCacheConfiguration, for the same reason that one does: this directory is
/// visible both to the configuration layer and to nes-physical-operators, whereas a header under
/// nes-physical-operators would drag that whole module into every consumer of WorkerConfiguration.

namespace NES
{

/// Where a slice's state should live. These are the leaves of the decision tree, ordered by how much
/// resident memory they give back: KeepInMemory gives back none, CompressInMemory gives back whatever
/// the compressor achieves, and the two spilling leaves give back all of it.
enum class StateDecision : uint8_t
{
    KeepInMemory,
    CompressInMemory,
    Spill,
    CompressAndSpill,
};

enum class CompressionAlgorithmType : uint8_t
{
    NONE,
    ZSTD,
};

enum class SpillStoreType : uint8_t
{
    IN_MEMORY,
    LOCAL_FILE,
};

/// Which StatePredictor a windowed operator uses to place its slices.
enum class StatePredictorType : uint8_t
{
    /// Measures what reduction costs at start-up and what the watermark is doing at runtime, then picks
    /// the leaf that frees the most memory while still fitting in the time available.
    COST_MODEL,
    /// Always takes the same branch, whatever the size or deadline. For benchmarking one leaf in
    /// isolation; also the in-memory baseline, as FORCED + KEEP_IN_MEMORY.
    FORCED,
};

/// Which WatermarkPredictor estimates when a slice will be needed.
enum class WatermarkPredictorType : uint8_t
{
    NONE,
    EWMA,
    KALMAN,
};

}
