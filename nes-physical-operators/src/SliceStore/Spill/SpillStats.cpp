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

#include <SliceStore/Spill/SpillStats.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace NES
{

SpillStatsRegistry& SpillStatsRegistry::instance()
{
    static SpillStatsRegistry registry;
    return registry;
}

SpillStatsRegistry::~SpillStatsRegistry()
{
    /// Stop the sampler before the registration map is destroyed — the loop reads it under the mutex.
    sampler.reset();
}

void SpillStatsRegistry::enableCsv(const std::string& path, const std::chrono::milliseconds interval)
{
    if (path.empty())
    {
        return;
    }
    const std::lock_guard lock{mutex};
    if (sampler != nullptr)
    {
        /// Already sampling. Every store calls this unconditionally, so repeat calls are expected.
        return;
    }
    csvPath = path;
    sampleInterval = interval.count() > 0 ? interval : std::chrono::milliseconds{100};
    sampler = std::make_unique<std::jthread>([this](const std::stop_token& stopToken) { samplerLoop(stopToken); });
}

uint64_t SpillStatsRegistry::registerStore(const Registration registration)
{
    const std::lock_guard lock{mutex};
    const auto storeId = nextStoreId++;
    registrations.emplace(storeId, registration);
    return storeId;
}

void SpillStatsRegistry::unregisterStore(const uint64_t storeId)
{
    const std::lock_guard lock{mutex};
    registrations.erase(storeId);
}

void SpillStatsRegistry::samplerLoop(const std::stop_token& stopToken) const
{
    std::ofstream file(csvPath);
    if (!file.is_open())
    {
        return;
    }
    file << "timestamp_ms,store_id,spills,restores,spill_failures,spilled_bytes,restored_bytes,"
            "kept_by_prediction,spilled_predictor_cold,spilled_beyond_horizon,"
            "spill_ns_total,restore_ns_total,restore_ns_max,resident_slices,spilled_slices\n";

    while (!stopToken.stop_requested())
    {
        std::this_thread::sleep_for(sampleInterval);
        if (stopToken.stop_requested())
        {
            break;
        }
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        /// Snapshot under the lock, format outside it: formatting must not block a store teardown.
        std::vector<std::pair<uint64_t, Registration>> snapshot;
        {
            const std::lock_guard lock{mutex};
            snapshot.assign(registrations.begin(), registrations.end());
        }
        for (const auto& [storeId, registration] : snapshot)
        {
            if (registration.stats == nullptr)
            {
                continue;
            }
            const auto& stats = *registration.stats;
            const auto policyStats = registration.policy != nullptr ? registration.policy->stats() : SpillPolicyStatistics{};
            file << now << ',' << storeId << ',' << stats.spills.load(std::memory_order_relaxed) << ','
                 << stats.restores.load(std::memory_order_relaxed) << ',' << stats.spillFailures.load(std::memory_order_relaxed) << ','
                 << stats.spilledBytes.load(std::memory_order_relaxed) << ','
                 << stats.restoredBytes.load(std::memory_order_relaxed) << ',' << policyStats.keptByPrediction << ','
                 << policyStats.spilledPredictorCold << ',' << policyStats.spilledBeyondHorizon << ','
                 << stats.spillNanosTotal.load(std::memory_order_relaxed) << ',' << stats.restoreNanosTotal.load(std::memory_order_relaxed)
                 << ',' << stats.restoreNanosMax.load(std::memory_order_relaxed) << ','
                 << stats.residentSlices.load(std::memory_order_relaxed) << ',' << stats.spilledSlices.load(std::memory_order_relaxed)
                 << '\n';
        }
        file.flush();
    }
}

}
