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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <SliceStore/Spill/SpillPolicy.hpp>

namespace NES
{

/// Counters a SpillingTimeBasedSliceStore maintains about its own I/O. Written from worker threads
/// on the GC/probe paths and read by the sampler thread, hence the atomics; relaxed ordering is
/// enough because no counter guards any other state.
struct SpillStatistics
{
    std::atomic<uint64_t> spills{0};
    std::atomic<uint64_t> restores{0};
    std::atomic<uint64_t> spillFailures{0};
    std::atomic<uint64_t> spilledBytes{0};
    std::atomic<uint64_t> restoredBytes{0};
    std::atomic<uint64_t> spillNanosTotal{0};
    std::atomic<uint64_t> restoreNanosTotal{0};
    std::atomic<uint64_t> restoreNanosMax{0};
    /// Gauges, overwritten once per GC tick rather than accumulated.
    std::atomic<uint64_t> residentSlices{0};
    std::atomic<uint64_t> spilledSlices{0};

    void addSpill(uint64_t bytes, uint64_t nanos) noexcept
    {
        spills.fetch_add(1, std::memory_order_relaxed);
        spilledBytes.fetch_add(bytes, std::memory_order_relaxed);
        spillNanosTotal.fetch_add(nanos, std::memory_order_relaxed);
    }

    void addRestore(uint64_t bytes, uint64_t nanos) noexcept
    {
        restores.fetch_add(1, std::memory_order_relaxed);
        restoredBytes.fetch_add(bytes, std::memory_order_relaxed);
        restoreNanosTotal.fetch_add(nanos, std::memory_order_relaxed);
        auto previousMax = restoreNanosMax.load(std::memory_order_relaxed);
        while (nanos > previousMax && !restoreNanosMax.compare_exchange_weak(previousMax, nanos, std::memory_order_relaxed))
        {
        }
    }
};

/// Process-global collection point. A worker runs many slice stores (one per stateful operator
/// instance), so counters are registered here and a single sampler thread writes one CSV row per
/// store per interval. Mirrors the buffer-usage monitor in BufferManager: enabled only when a path
/// is configured, and a no-op otherwise.
class SpillStatsRegistry
{
public:
    static SpillStatsRegistry& instance();

    SpillStatsRegistry(const SpillStatsRegistry&) = delete;
    SpillStatsRegistry(SpillStatsRegistry&&) = delete;
    SpillStatsRegistry& operator=(const SpillStatsRegistry&) = delete;
    SpillStatsRegistry& operator=(SpillStatsRegistry&&) = delete;

    /// Both pointers are owned by the registering store and must outlive its unregisterStore call.
    struct Registration
    {
        const SpillStatistics* stats{nullptr};
        const SpillPolicy* policy{nullptr};
    };

    /// Starts the sampler on the first call with a non-empty path; later calls with the same path
    /// are ignored, so every store may call it unconditionally.
    void enableCsv(const std::string& path, std::chrono::milliseconds interval);

    [[nodiscard]] uint64_t registerStore(Registration registration);
    void unregisterStore(uint64_t storeId);

private:
    SpillStatsRegistry() = default;
    ~SpillStatsRegistry();

    void samplerLoop(const std::stop_token& stopToken) const;

    mutable std::mutex mutex;
    std::unordered_map<uint64_t, Registration> registrations;
    uint64_t nextStoreId{1};
    std::string csvPath;
    std::chrono::milliseconds sampleInterval{100};
    std::unique_ptr<std::jthread> sampler;
};

}
