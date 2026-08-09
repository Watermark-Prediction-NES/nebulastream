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
#include <functional>
#include <memory>
#include <vector>
#include <Time/Timestamp.hpp>

namespace NES
{
using SliceStart = Timestamp;
using SliceEnd = Timestamp;

struct CreateNewSlicesArguments
{
    virtual ~CreateNewSlicesArguments() = default;
};

/// This enum helps to keep track of the status of a window by classifying the stages of the join
/// The state transitions are as follows:
///                Current State |     Action                        | Next State
/// ----------------------------------------------------------------------------------------------
/// Start                        | Create Slice                      | WINDOW_FILLING
/// WINDOW_FILLING               | Global Watermark Ts > WindowEnd   | EMITTED_TO_PROBE
/// WINDOW_FILLING               | Left or Right Pipeline Terminated | WAITING_ON_TERMINATION
/// WAITING_ON_TERMINATION       | Query/Pipeline Terminated         | EMITTED_TO_PROBE
/// EMITTED_TO_PROBE             | Tuples join in Probe              | CAN_BE_DELETED
enum class WindowInfoState : uint8_t
{
    WINDOW_FILLING,
    WAITING_ON_TERMINATION,
    EMITTED_TO_PROBE
};

/// This class represents a single slice
class Slice
{
public:
    Slice(SliceStart sliceStart, SliceEnd sliceEnd);
    Slice(const Slice& other);
    Slice(Slice&& other) noexcept;
    Slice& operator=(const Slice& other);
    Slice& operator=(Slice&& other) noexcept;
    virtual ~Slice() = default;

    [[nodiscard]] SliceStart getSliceStart() const;
    [[nodiscard]] SliceEnd getSliceEnd() const;

    /// Gives a retired slice a new identity so it can be reused for a different time range.
    /// Only a slice-creation function may call this, and only for slices that no probe can reach anymore.
    void reassign(SliceStart newSliceStart, SliceEnd newSliceEnd);

    /// Empties the slice's data structures in place so it can serve a new time range, keeping allocated
    /// memory. Returns false if this slice type cannot be reset (it must be destroyed instead).
    [[nodiscard]] virtual bool resetForReuse() { return false; }

    bool operator==(const Slice& rhs) const;
    bool operator!=(const Slice& rhs) const;

protected:
    SliceStart sliceStart;
    SliceEnd sliceEnd;
};

/// Creates the slices for [sliceStart, sliceEnd). recycledCandidate may hold a retired slice from the
/// slice store's pool; the function may reassign and return it instead of allocating, if and only if it
/// structurally matches what the function would have created. A rejected candidate is simply dropped.
using SliceCreateFunction = std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd, const std::shared_ptr<Slice>&)>;
}
