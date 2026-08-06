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

#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <unordered_map>
#include <folly/Synchronized.h>
#include <QueryEngineStatisticListener.hpp>
#include <QueryId.hpp>

namespace NES
{

/// Reports a per-query tuple rate at a fixed interval. This is the throughput half of the
/// state-reduction measurements; the memory half comes from the BufferManager's buffer-usage CSV.
///
/// What it counts is every tuple handed from one pipeline to the next, summed over a query's pipelines.
/// A tuple that passes through three pipelines therefore counts three times, so the absolute number is
/// not "tuples per second through the query" and should not be read as one. It is a fixed multiple of
/// that, and the multiple is a property of the plan — which makes it directly comparable across runs of
/// the same query, which is exactly what a state-reduction benchmark does.
///
/// Counting the source's own output instead would be the more natural metric, but the engine emits no
/// event for it: a source feeding the first pipeline produces no TaskEmit.
class ThroughputListener final : public QueryEngineStatisticListener
{
public:
    struct Report
    {
        QueryId queryId;
        std::chrono::milliseconds interval{};
        uint64_t tuples{};
        double tuplesPerSecond{};
    };

    /// @param interval how often the callback fires, per query with activity in that interval
    /// @param report invoked from the listener's own thread, never from a worker thread
    ThroughputListener(std::chrono::milliseconds interval, std::function<void(const Report&)> report);
    ~ThroughputListener() override;

    ThroughputListener(const ThroughputListener&) = delete;
    ThroughputListener(ThroughputListener&&) = delete;
    ThroughputListener& operator=(const ThroughputListener&) = delete;
    ThroughputListener& operator=(ThroughputListener&&) = delete;

    void onEvent(Event event) override;

private:
    /// Accumulate-and-drain rather than a queue of events: onEvent runs on every worker thread on the
    /// task path, so it does as little as possible — one lock and one addition — and all the arithmetic
    /// happens on the reporting thread.
    folly::Synchronized<std::unordered_map<QueryId, uint64_t>> tuplesPerQuery;
    std::chrono::milliseconds interval;
    std::function<void(const Report&)> report;
    std::jthread reportingThread;
};

}
