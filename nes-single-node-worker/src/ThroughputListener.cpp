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

#include <ThroughputListener.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <Util/Overloaded.hpp>
#include <ErrorHandling.hpp>
#include <QueryEngineStatisticListener.hpp>
#include <QueryId.hpp>
#include <Thread.hpp>

namespace NES
{

ThroughputListener::ThroughputListener(const std::chrono::milliseconds interval, std::function<void(const Report&)> report)
    : interval(interval), report(std::move(report))
{
    PRECONDITION(interval.count() > 0, "Reporting interval must be positive, but is {} ms", interval.count());
    PRECONDITION(this->report != nullptr, "Reporting callback must not be null");

    reportingThread = std::jthread(
        [this](const std::stop_token& stopToken)
        {
            Thread::setThreadName("ThroughputListener");
            while (not stopToken.stop_requested())
            {
                std::this_thread::sleep_for(this->interval);

                /// Drained rather than read: each report covers one interval, so a query with no activity
                /// simply produces no report instead of repeating its last one.
                std::unordered_map<QueryId, uint64_t> drained;
                tuplesPerQuery.withWLock([&drained](auto& counts) { drained.swap(counts); });

                const auto seconds = std::chrono::duration<double>(this->interval).count();
                for (const auto& [queryId, tuples] : drained)
                {
                    this->report(Report{
                        .queryId = queryId,
                        .interval = this->interval,
                        .tuples = tuples,
                        .tuplesPerSecond = static_cast<double>(tuples) / seconds});
                }
            }
        });
}

ThroughputListener::~ThroughputListener() = default;

void ThroughputListener::onEvent(Event event)
{
    /// Runs on a worker thread on the task path, so everything except the addition is left to the
    /// reporting thread.
    std::visit(
        Overloaded{
            [this](const TaskEmit& taskEmit)
            { tuplesPerQuery.withWLock([&taskEmit](auto& counts) { counts[taskEmit.queryId] += taskEmit.numberOfTuples; }); },
            [](const auto&) {}},
        event);
}

}
