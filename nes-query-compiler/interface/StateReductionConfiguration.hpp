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

#include <memory>
#include <string>
#include <vector>
#include <Configurations/BaseConfiguration.hpp>
#include <Configurations/BaseOption.hpp>
#include <Configurations/Enums/EnumOption.hpp>
#include <Configurations/ScalarOption.hpp>
#include <Configurations/Validation/FloatValidation.hpp>
#include <Configurations/Validation/NumberValidation.hpp>
#include <fmt/format.h>
#include <StateReductionTypes.hpp>

namespace NES
{

class StateReductionConfiguration final : public BaseConfiguration
{
public:
    StateReductionConfiguration() = default;
    StateReductionConfiguration(const std::string& name, const std::string& description) : BaseConfiguration(name, description) { };

    BoolOption enabled
        = {"enabled",
           "false",
           "Whether windowed operators may move slice state out of resident memory by compressing it, "
           "spilling it, or both."};

    EnumOption<StatePredictorType> predictor
        = {"predictor",
           StatePredictorType::COST_MODEL,
           fmt::format("How the placement of a slice is decided: {}", enumPipeList<StatePredictorType>())};

    EnumOption<StateDecision> forcedDecision
        = {"forced_decision",
           StateDecision::KeepInMemory,
           fmt::format("Which branch predictor=FORCED always takes; ignored otherwise: {}", enumPipeList<StateDecision>())};

    EnumOption<WatermarkPredictorType> watermarkPredictor
        = {"watermark_predictor",
           WatermarkPredictorType::EWMA,
           fmt::format(
               "Estimates when a slice will next be needed, which is what the cost model spends. NONE leaves every "
               "slice looking immediately needed, so nothing is ever reduced: {}",
               enumPipeList<WatermarkPredictorType>())};

    EnumOption<CompressionAlgorithmType> compression
        = {"compression",
           CompressionAlgorithmType::ZSTD,
           fmt::format("Compressor for the compressing branches: {}", enumPipeList<CompressionAlgorithmType>())};

    UIntOption compressionLevel
        = {"compression_level", "3", "Effort level passed to the compressor.", {std::make_shared<NumberValidation>()}};

    EnumOption<SpillStoreType> spillStore
        = {"spill_store", SpillStoreType::LOCAL_FILE, fmt::format("Where spilled state is parked: {}", enumPipeList<SpillStoreType>())};

    StringOption spillDirectory
        = {"spill_directory",
           "/tmp/nes-spill",
           "Root for LOCAL_FILE spilling. Each operator gets its own subdirectory, which it removes when it stops."};

    UIntOption memoryTargetBytes
        = {"operator_memory_target_bytes",
           "0",
           "Soft budget per windowed operator. Below it nothing is reduced, however cheap a reduction looks; above it "
           "the predictor decides. 0 disables the budget, leaving the predictor in charge at every size.",
           {std::make_shared<NumberValidation>()}};

    UIntOption memoryCeilingBytes
        = {"operator_memory_ceiling_bytes",
           "0",
           "Hard budget per windowed operator. Above it slices are reduced as aggressively as possible regardless of "
           "predicted cost, worst-predicted-need first, until resident bytes are back under. 0 disables the ceiling.",
           {std::make_shared<NumberValidation>()}};

    UIntOption calibrationRepetitions
        = {"calibration_repetitions",
           "3",
           "How many times each payload size is measured during start-up calibration. Higher is steadier and slower.",
           {std::make_shared<NumberValidation>()}};

    StringOption statsLogPath
        = {"stats_log_path",
           "",
           "CSV file that each windowed operator appends one summary row to when it stops: what it "
           "decided, how much it moved, and how often a slice came back sooner than moving it cost. Empty "
           "disables. The file is appended to, so a whole benchmark sweep can share one path and be "
           "plotted from it."};

    FloatOption learningRate
        = {"learning_rate",
           "0.2",
           "How much weight a runtime observation gets when correcting the calibrated cost model, in [0, 1]. 0 freezes "
           "the calibration.",
           {std::make_shared<FloatValidation>(0.0, 1.0)}};

private:
    std::vector<BaseOption*> getOptions() override
    {
        return {
            &enabled,
            &predictor,
            &forcedDecision,
            &watermarkPredictor,
            &compression,
            &compressionLevel,
            &spillStore,
            &spillDirectory,
            &memoryTargetBytes,
            &memoryCeilingBytes,
            &calibrationRepetitions,
            &statsLogPath,
            &learningRate};
    }
};

}
