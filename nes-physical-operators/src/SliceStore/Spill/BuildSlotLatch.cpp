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

#include <SliceStore/Spill/BuildSlotLatch.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace NES
{

namespace
{
/// Short spin before yielding: a build task is typically microseconds, so the barrier usually clears
/// almost immediately and a full sleep would dominate.
void backoff(uint32_t& spins) noexcept
{
    if (spins < 64)
    {
        ++spins;
        std::this_thread::yield();
        return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds{100});
}
}

BuildSlotLatch::BuildSlotLatch(const uint64_t numberOfWorkerThreads) : slots(numberOfWorkerThreads == 0 ? 1 : numberOfWorkerThreads)
{
}

bool BuildSlotLatch::enterBuild(const WorkerThreadId workerThreadId) noexcept
{
    const auto slot = workerThreadId.getRawValue() % slots.size();

    const auto deadline = std::chrono::steady_clock::now() + BuilderWaitTimeout;
    uint32_t spins = 0;
    while (true)
    {
        /// Publish busy first, then re-check the barrier. Doing it in this order (rather than
        /// check-then-set) is what closes the window where a spiller observes an idle slot at the
        /// same moment a builder decides to enter: if the spiller raised the barrier after our store,
        /// it must observe busy==true and wait for us; if it raised it before, we see it here and
        /// back out. Both atomics are seq_cst so the two stores cannot be reordered past each other.
        slots[slot].busy.store(true, std::memory_order_seq_cst);
        if (!barrierRaised.load(std::memory_order_seq_cst))
        {
            return true;
        }
        slots[slot].busy.store(false, std::memory_order_seq_cst);

        if (std::chrono::steady_clock::now() >= deadline)
        {
            /// A spiller is stuck. Refuse the task rather than write into a structure that may be
            /// mid-spill; the caller treats this as "cannot use the state this buffer".
            return false;
        }
        backoff(spins);
    }
}

void BuildSlotLatch::exitBuild(const WorkerThreadId workerThreadId) noexcept
{
    slots[workerThreadId.getRawValue() % slots.size()].busy.store(false, std::memory_order_seq_cst);
}

bool BuildSlotLatch::acquireBarrier() noexcept
{
    bool expected = false;
    if (!barrierRaised.compare_exchange_strong(expected, true, std::memory_order_seq_cst))
    {
        /// Another spiller already holds it. Only one GC tick spills at a time; do not queue.
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() + BarrierAcquireTimeout;
    uint32_t spins = 0;
    for (auto& slot : slots)
    {
        while (slot.busy.load(std::memory_order_seq_cst))
        {
            if (std::chrono::steady_clock::now() >= deadline)
            {
                /// Someone is taking too long, or a task died without clearing its slot. Lower the
                /// barrier so builders are not held up, and tell the caller not to spill.
                barrierRaised.store(false, std::memory_order_seq_cst);
                return false;
            }
            backoff(spins);
        }
    }
    return true;
}

void BuildSlotLatch::releaseBarrier() noexcept
{
    barrierRaised.store(false, std::memory_order_seq_cst);
}

}
