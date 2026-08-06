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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include <StateReduction/CompressionAlgorithm.hpp>
#include <StateReduction/CostModelStatePredictor.hpp>
#include <StateReduction/ForcedStatePredictor.hpp>
#include <StateReduction/SpillStore.hpp>
#include <StateReduction/StateReductionCalibrator.hpp>
#include <StateReduction/StateReductionCostModel.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <BaseUnitTest.hpp>
#include <ErrorHandling.hpp>
#include <StateReductionTypes.hpp>

namespace NES
{
namespace
{
/// A model with round numbers, so the expected decision in each test can be worked out by hand rather
/// than by running the code under test.
///   compress   1 ns/byte, decompress 1 ns/byte
///   spillWrite 10 ns/byte, spillRead 10 ns/byte
///   ratio      0.5
/// For 1,000,000 bytes that gives round trips of:
///   KeepInMemory     0 ms
///   CompressInMemory 1 ms compress + 0.5 ms decompress            = 1.5 ms
///   CompressAndSpill 1 ms compress + 5 ms write + 5 ms read + 0.5 = 11.5 ms
///   Spill            10 ms write + 10 ms read                     = 20 ms
StateReductionCostModel roundNumberModel()
{
    StateReductionCostModel model;
    model.compress = {.fixedNanos = 0.0, .nanosPerByte = 1.0};
    model.decompress = {.fixedNanos = 0.0, .nanosPerByte = 1.0};
    model.spillWrite = {.fixedNanos = 0.0, .nanosPerByte = 10.0};
    model.spillRead = {.fixedNanos = 0.0, .nanosPerByte = 10.0};
    model.compressionRatio = 0.5;
    return model;
}

constexpr uint64_t OneMegabyte = 1'000'000;

std::vector<std::byte> compressiblePayload(const size_t bytes)
{
    std::vector<std::byte> payload(bytes);
    for (size_t i = 0; i < bytes; ++i)
    {
        payload[i] = static_cast<std::byte>(i % 16);
    }
    return payload;
}
}

class StateReductionTest : public Testing::BaseUnitTest
{
public:
    static void SetUpTestSuite() { Logger::setupLogging("StateReductionTest.log", LogLevel::LOG_DEBUG); }

    void SetUp() override
    {
        Testing::BaseUnitTest::SetUp();
        /// Per-test root, so the LOCAL_FILE cases cannot collide when ctest runs the suite in parallel.
        const auto* const testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
        spillDirectory = std::filesystem::temp_directory_path() / "nes-state-reduction-test"
            / (std::string{testInfo->test_suite_name()} + "-" + testInfo->name());
    }

    void TearDown() override
    {
        std::error_code errorCode;
        std::ignore = std::filesystem::remove_all(spillDirectory, errorCode);
        Testing::BaseUnitTest::TearDown();
    }

    std::filesystem::path spillDirectory;
};

/// ---------------------------------------------------------------------------------------------
/// StatePredictor
/// ---------------------------------------------------------------------------------------------

TEST_F(StateReductionTest, WithNoTimeAtAllNothingIsReduced)
{
    const CostModelStatePredictor predictor{roundNumberModel(), 0.2};

    /// A cold watermark predictor reports a zero budget. Only KeepInMemory costs nothing, so this is
    /// what keeps an unwarmed system from reducing state it cannot get back.
    EXPECT_EQ(predictor.predict(OneMegabyte, 0), StateDecision::KeepInMemory);
}

TEST_F(StateReductionTest, WithATinyBudgetNothingIsReduced)
{
    const CostModelStatePredictor predictor{roundNumberModel(), 0.2};

    /// 1 ms is below the 1.5 ms round trip of the cheapest reduction.
    EXPECT_EQ(predictor.predict(OneMegabyte, 1), StateDecision::KeepInMemory);
}

TEST_F(StateReductionTest, WithEnoughTimeToCompressButNotToSpillItCompresses)
{
    const CostModelStatePredictor predictor{roundNumberModel(), 0.2};

    /// 5 ms clears CompressInMemory's 1.5 ms but not CompressAndSpill's 11.5 ms.
    EXPECT_EQ(predictor.predict(OneMegabyte, 5), StateDecision::CompressInMemory);
}

TEST_F(StateReductionTest, WithAmpleTimeItPicksTheLeafThatFreesEverythingMostCheaply)
{
    const CostModelStatePredictor predictor{roundNumberModel(), 0.2};

    /// At 15 ms both CompressAndSpill (11.5 ms) and CompressInMemory (1.5 ms) fit, but only the former
    /// frees everything. Spill also frees everything and does not fit yet.
    EXPECT_EQ(predictor.predict(OneMegabyte, 15), StateDecision::CompressAndSpill);

    /// At 50 ms Spill fits too. It frees exactly as much as CompressAndSpill, so the tie is broken on
    /// cost, and compressing first is cheaper than writing twice the bytes.
    EXPECT_EQ(predictor.predict(OneMegabyte, 50), StateDecision::CompressAndSpill);
}

TEST_F(StateReductionTest, WhenCompressionBuysNothingSpillingWins)
{
    auto model = roundNumberModel();
    /// Incompressible data: compressing costs CPU and saves no bytes, so CompressAndSpill degenerates
    /// into Spill plus wasted work and the tie-break should prefer plain Spill.
    model.compressionRatio = 1.0;
    const CostModelStatePredictor predictor{model, 0.2};

    EXPECT_EQ(predictor.predict(OneMegabyte, 100), StateDecision::Spill);
}

TEST_F(StateReductionTest, CompressingIsNotWorthItWhenItFreesNothing)
{
    auto model = roundNumberModel();
    model.compressionRatio = 1.0;
    const CostModelStatePredictor predictor{model, 0.2};

    /// The only case where two decisions leave the same bytes resident. At 5 ms neither spilling leaf
    /// fits (20 ms and 22 ms), so the choice is between KeepInMemory and a compression that costs 2 ms
    /// and reclaims nothing. Paying for a reduction that frees no memory is never right, so the
    /// resident-then-cost ordering has to land on KeepInMemory here.
    EXPECT_EQ(predictor.predict(OneMegabyte, 5), StateDecision::KeepInMemory);
}

TEST_F(StateReductionTest, SmallStateIsCheapEnoughToSpillOutright)
{
    const CostModelStatePredictor predictor{roundNumberModel(), 0.2};

    /// 1 KiB spills in 20 us, so even a 1 ms budget covers the leaf that frees everything.
    EXPECT_EQ(predictor.predict(1024, 1), StateDecision::CompressAndSpill);
}

TEST_F(StateReductionTest, ObservedCompressionRatioReplacesTheCalibratedGuess)
{
    CostModelStatePredictor predictor{roundNumberModel(), 1.0};
    ASSERT_DOUBLE_EQ(predictor.currentModel().compressionRatio, 0.5);

    /// A learning rate of 1 takes the observation wholesale: 90% freed means a ratio of 0.1.
    predictor.update(StateDecision::CompressInMemory, 0.9, OneMegabyte, std::chrono::milliseconds{1});

    EXPECT_NEAR(predictor.currentModel().compressionRatio, 0.1, 1e-9);
}

TEST_F(StateReductionTest, SpillObservationsDoNotTouchTheCompressionRatio)
{
    CostModelStatePredictor predictor{roundNumberModel(), 1.0};

    /// Spill stores state verbatim, so it frees everything resident but says nothing about how well the
    /// data compresses. Letting it move the ratio would teach the model that everything compresses
    /// perfectly.
    predictor.update(StateDecision::Spill, 1.0, OneMegabyte, std::chrono::milliseconds{10});

    EXPECT_DOUBLE_EQ(predictor.currentModel().compressionRatio, 0.5);
}

TEST_F(StateReductionTest, ASlowerThanPredictedReductionMakesTheModelMorePessimistic)
{
    CostModelStatePredictor predictor{roundNumberModel(), 1.0};
    const auto before = predictor.currentModel().spillWrite.nanosPerByte;

    /// Predicted 10 ms for a 1 MB spill; it actually took 20 ms.
    predictor.update(StateDecision::Spill, 1.0, OneMegabyte, std::chrono::milliseconds{20});

    EXPECT_GT(predictor.currentModel().spillWrite.nanosPerByte, before);
}

TEST_F(StateReductionTest, ARestoreObservationCorrectsTheReadSideOnly)
{
    CostModelStatePredictor predictor{roundNumberModel(), 1.0};
    const auto writeBefore = predictor.currentModel().spillWrite.nanosPerByte;

    predictor.updateAfterRestore(StateDecision::Spill, OneMegabyte, std::chrono::milliseconds{30});

    EXPECT_GT(predictor.currentModel().spillRead.nanosPerByte, 10.0);
    EXPECT_DOUBLE_EQ(predictor.currentModel().spillWrite.nanosPerByte, writeBefore);
}

TEST_F(StateReductionTest, OneWildTimingCannotRunAwayWithTheModel)
{
    CostModelStatePredictor predictor{roundNumberModel(), 1.0};

    /// A timing that caught a stop-the-world pause. Believing it outright would push the model so far
    /// that it stops reducing anything, and it would then never see another observation to recover from.
    predictor.update(StateDecision::Spill, 1.0, OneMegabyte, std::chrono::seconds{10});

    EXPECT_LE(predictor.currentModel().spillWrite.nanosPerByte, 10.0 * 10.0);
}

TEST_F(StateReductionTest, ForcedPredictorIgnoresSizeAndDeadline)
{
    const ForcedStatePredictor predictor{StateDecision::CompressAndSpill};

    EXPECT_EQ(predictor.predict(0, 0), StateDecision::CompressAndSpill);
    EXPECT_EQ(predictor.predict(OneMegabyte, 0), StateDecision::CompressAndSpill);
    EXPECT_EQ(predictor.predict(1, 1'000'000), StateDecision::CompressAndSpill);
}

/// ---------------------------------------------------------------------------------------------
/// CompressionAlgorithm — one body, both implementations
/// ---------------------------------------------------------------------------------------------

class CompressionAlgorithmContractTest : public StateReductionTest, public ::testing::WithParamInterface<CompressionAlgorithmType>
{
};

TEST_P(CompressionAlgorithmContractTest, RoundTripsPayloadsOfEverySize)
{
    const auto compression = CompressionAlgorithm::create(GetParam(), 3);

    for (const size_t size : {size_t{0}, size_t{1}, size_t{17}, size_t{4096}, size_t{100'000}})
    {
        const auto payload = compressiblePayload(size);
        const auto compressed = compression->compress(payload);
        const auto restored = compression->decompress(compressed, size);
        EXPECT_EQ(restored, payload) << "round trip failed at size " << size;
    }
}

INSTANTIATE_TEST_CASE_P(
    AllAlgorithms, CompressionAlgorithmContractTest, ::testing::Values(CompressionAlgorithmType::NONE, CompressionAlgorithmType::ZSTD));

TEST_F(StateReductionTest, ZstdActuallyShrinksCompressibleState)
{
    const auto compression = CompressionAlgorithm::create(CompressionAlgorithmType::ZSTD, 3);
    const auto payload = compressiblePayload(100'000);

    /// If this ever stops holding, the whole CompressInMemory leaf is pointless and the tests above that
    /// assume a ratio below 1 are testing a fiction.
    EXPECT_LT(compression->compress(payload).size(), payload.size());
}

TEST_F(StateReductionTest, NoCompressionLeavesBytesUntouched)
{
    const auto compression = CompressionAlgorithm::create(CompressionAlgorithmType::NONE, 3);
    const auto payload = compressiblePayload(1024);

    EXPECT_EQ(compression->compress(payload), payload);
}

/// ---------------------------------------------------------------------------------------------
/// SpillStore — one body, both implementations
/// ---------------------------------------------------------------------------------------------

class SpillStoreContractTest : public StateReductionTest, public ::testing::WithParamInterface<SpillStoreType>
{
};

TEST_P(SpillStoreContractTest, RoundTripsStoredState)
{
    const auto store = SpillStore::create(GetParam(), spillDirectory / "roundtrip");
    const auto payload = compressiblePayload(4096);

    store->put(42, payload);
    EXPECT_EQ(store->get(42), payload);
}

TEST_P(SpillStoreContractTest, PutReplacesWhatWasThere)
{
    const auto store = SpillStore::create(GetParam(), spillDirectory / "replace");
    store->put(7, compressiblePayload(4096));

    const auto replacement = compressiblePayload(64);
    store->put(7, replacement);

    EXPECT_EQ(store->get(7), replacement);
    EXPECT_EQ(store->storedBytes(), replacement.size()) << "the replaced object must stop being accounted for";
}

TEST_P(SpillStoreContractTest, KeepsSlicesApart)
{
    const auto store = SpillStore::create(GetParam(), spillDirectory / "distinct");
    const auto first = compressiblePayload(128);
    const auto second = compressiblePayload(256);

    store->put(1, first);
    store->put(2, second);

    EXPECT_EQ(store->get(1), first);
    EXPECT_EQ(store->get(2), second);
}

TEST_P(SpillStoreContractTest, ReadingAnUnknownSliceThrows)
{
    const auto store = SpillStore::create(GetParam(), spillDirectory / "missing");

    ASSERT_EXCEPTION_ERRORCODE(std::ignore = store->get(999), ErrorCode::CannotDeserialize);
}

TEST_P(SpillStoreContractTest, ErasingASliceThatWasNeverStoredIsFine)
{
    const auto store = SpillStore::create(GetParam(), spillDirectory / "erase-missing");

    /// Garbage collection erases every slice it discards without tracking which ones were reduced.
    EXPECT_NO_THROW(store->erase(123));
    EXPECT_EQ(store->storedBytes(), 0U);
}

TEST_P(SpillStoreContractTest, StoredBytesTracksWhatIsHeld)
{
    const auto store = SpillStore::create(GetParam(), spillDirectory / "accounting");
    EXPECT_EQ(store->storedBytes(), 0U);

    store->put(1, compressiblePayload(1000));
    store->put(2, compressiblePayload(500));
    EXPECT_EQ(store->storedBytes(), 1500U);

    store->erase(1);
    EXPECT_EQ(store->storedBytes(), 500U);
}

TEST_P(SpillStoreContractTest, RoundTripsAnEmptyPayload)
{
    const auto store = SpillStore::create(GetParam(), spillDirectory / "empty");
    store->put(1, {});

    EXPECT_TRUE(store->get(1).empty());
}

INSTANTIATE_TEST_CASE_P(AllStores, SpillStoreContractTest, ::testing::Values(SpillStoreType::IN_MEMORY, SpillStoreType::LOCAL_FILE));

TEST_F(StateReductionTest, LocalFileStoreCleansUpItsDirectory)
{
    const auto directory = spillDirectory / "cleanup";
    {
        const auto store = SpillStore::create(SpillStoreType::LOCAL_FILE, directory);
        store->put(1, compressiblePayload(1024));
        ASSERT_TRUE(std::filesystem::exists(directory));
    }

    /// Spilled state outliving the query that produced it fills the disk of a long-running worker.
    EXPECT_FALSE(std::filesystem::exists(directory));
}

/// ---------------------------------------------------------------------------------------------
/// Calibration
/// ---------------------------------------------------------------------------------------------

TEST_F(StateReductionTest, CalibrationProducesAUsableModel)
{
    const auto compression = CompressionAlgorithm::create(CompressionAlgorithmType::ZSTD, 3);
    const auto store = SpillStore::create(SpillStoreType::IN_MEMORY, spillDirectory / "calibrate");

    const auto model = calibrateStateReduction(*compression, *store, CalibrationConfig{.maxBytes = 65536, .repetitions = 1});

    /// Not asserting on absolute timings — those are the machine's business. What must hold is that the
    /// model is well formed, because a zero or negative coefficient would make a reduction look free and
    /// the predictor would choose it regardless of the deadline.
    EXPECT_GT(model.compress.nanosPerByte, 0.0);
    EXPECT_GT(model.decompress.nanosPerByte, 0.0);
    EXPECT_GE(model.spillWrite.nanosPerByte, 0.0);
    EXPECT_GE(model.spillRead.nanosPerByte, 0.0);
    EXPECT_GT(model.compressionRatio, 0.0);
    EXPECT_LE(model.compressionRatio, 1.0);
}

TEST_F(StateReductionTest, CalibrationLeavesNothingBehindInTheStore)
{
    const auto compression = CompressionAlgorithm::create(CompressionAlgorithmType::ZSTD, 3);
    const auto store = SpillStore::create(SpillStoreType::IN_MEMORY, spillDirectory / "calibrate-clean");

    std::ignore = calibrateStateReduction(*compression, *store, CalibrationConfig{.maxBytes = 65536, .repetitions = 1});

    EXPECT_EQ(store->storedBytes(), 0U) << "calibration payloads must not be mistaken for real slice state";
}

TEST_F(StateReductionTest, CalibrationRecognisesIncompressibleData)
{
    const auto compression = CompressionAlgorithm::create(CompressionAlgorithmType::NONE, 3);
    const auto store = SpillStore::create(SpillStoreType::IN_MEMORY, spillDirectory / "calibrate-none");

    const auto model = calibrateStateReduction(*compression, *store, CalibrationConfig{.maxBytes = 65536, .repetitions = 1});

    /// With compression disabled the ratio must be 1, otherwise the predictor would believe
    /// CompressInMemory frees memory that it does not.
    EXPECT_DOUBLE_EQ(model.compressionRatio, 1.0);
}
}
