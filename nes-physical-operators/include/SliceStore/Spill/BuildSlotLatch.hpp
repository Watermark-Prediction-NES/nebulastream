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
#include <vector>
#include <Identifiers/Identifiers.hpp>

namespace NES
{

/// Mutual exclusion between the JIT-compiled build path and the spill path of one window operator.
///
/// WHY THIS EXISTS. A build task writes into a slice's per-worker-thread data structure through a raw
/// pointer cached in a per-pipeline SliceCache, entirely inside JIT-compiled code and without taking
/// any lock. Spilling mutates those same structures (HJSliceStateSerializer calls
/// ChainedHashMap::clear, NLJ drains the PagedVector's pages). Spilling a slice that a build task may
/// still be writing to therefore segfaults inside ChainedHashMap::insertEntry. Slices at or below the
/// global watermark are safe without any of this — no build tuple can be assigned to them any more —
/// so this latch is only needed to spill slices that are still filling, which is where most of the
/// memory actually sits.
///
/// SHAPE. Builders mark themselves busy for the duration of one task (one TupleBuffer), not one
/// tuple, so the cost is one atomic store per buffer on the hot path. The spiller raises a barrier
/// that keeps new tasks out, waits for the in-flight ones to finish, spills, and lowers it.
///
/// FAILURE BEHAVIOUR IS DEGRADATION, NEVER DEADLOCK. Every wait is bounded:
///   - if a builder's task dies between enterBuild() and exitBuild() its slot stays marked busy, so
///     acquireBarrier() times out and the caller simply does not spill that slice this tick. The
///     flag is cleared again by that thread's next exitBuild().
///   - builders never block longer than the barrier timeout either, so a spiller that is somehow
///     stuck cannot stall ingestion indefinitely.
class BuildSlotLatch
{
public:
    /// Waits at most this long for in-flight build tasks to drain before giving up on a spill.
    static constexpr std::chrono::milliseconds BarrierAcquireTimeout{20};
    /// A builder waits at most this long for a barrier to lift before proceeding anyway. Exceeding it
    /// means a spiller is stuck; letting the builder through risks the race we are trying to avoid,
    /// so the builder instead reports that it could not enter and the caller skips the slice cache
    /// fast path — see enterBuild()'s return value.
    static constexpr std::chrono::milliseconds BuilderWaitTimeout{100};

    explicit BuildSlotLatch(uint64_t numberOfWorkerThreads);

    /// Marks this worker thread as inside a build task. Returns false only if a barrier stayed up
    /// past BuilderWaitTimeout, in which case the caller must NOT write to any slice.
    bool enterBuild(WorkerThreadId workerThreadId) noexcept;
    void exitBuild(WorkerThreadId workerThreadId) noexcept;

    /// Blocks new build tasks and waits for the in-flight ones. Returns false on timeout, leaving the
    /// barrier lowered — the caller must then skip the spill rather than proceed unsafely.
    [[nodiscard]] bool acquireBarrier() noexcept;
    void releaseBarrier() noexcept;

    /// RAII wrapper; `held()` reports whether the barrier was actually acquired.
    class Barrier
    {
    public:
        explicit Barrier(BuildSlotLatch* latch) : latch(latch), acquired(latch != nullptr && latch->acquireBarrier()) { }
        ~Barrier()
        {
            if (acquired)
            {
                latch->releaseBarrier();
            }
        }
        Barrier(const Barrier&) = delete;
        Barrier(Barrier&&) = delete;
        Barrier& operator=(const Barrier&) = delete;
        Barrier& operator=(Barrier&&) = delete;

        [[nodiscard]] bool held() const noexcept { return acquired; }

    private:
        BuildSlotLatch* latch;
        bool acquired;
    };

    [[nodiscard]] uint64_t numberOfSlots() const noexcept { return slots.size(); }

private:
    /// One cache-line-padded flag per worker thread. Padding matters: builders on different threads
    /// store to their own slot on every buffer, and false sharing here would show up as a throughput
    /// regression on the operator's hot path.
    struct alignas(64) Slot
    {
        std::atomic<bool> busy{false};
    };

    std::vector<Slot> slots;
    std::atomic<bool> barrierRaised{false};
};

}
