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
#include <HashMapSlice.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <Identifiers/Identifiers.hpp>
#include <Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Interface/HashMap/HashMap.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SliceStore/Slice.hpp>
#include <StateReduction/ReducibleSlice.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

HashMapSlice::HashMapSlice(
    SliceStart sliceStart,
    SliceEnd sliceEnd,
    const CreateNewHashMapSliceArgs& createNewHashMapSliceArgs,
    const uint64_t numberOfHashMaps,
    const uint64_t numberOfInputStreams)
    : Slice(sliceStart, sliceEnd)
    , numHashMaps(numberOfHashMaps * numberOfInputStreams)
    , numInputStreams(numberOfInputStreams)
    , numHashmapsPerInputStream(numberOfHashMaps)
    , params(createNewHashMapSliceArgs)
{
    hashMapBuffers.resize(numHashMaps);
    hashMapBuffersState.assign(numHashMaps, HashMapBufferState::UNINITIALIZED);
}

uint64_t HashMapSlice::getNumberOfHashMaps() const
{
    return numHashMaps;
}

uint64_t HashMapSlice::getNumInputStreams() const
{
    return numInputStreams;
}

uint64_t HashMapSlice::getNumHashMapsPerInputStream() const
{
    return numHashmapsPerInputStream;
}

const TupleBuffer* HashMapSlice::getHashMapBufferRef(const ChildBufferIndex childBufferIndex) const
{
    PRECONDITION(childBufferIndex.getRawValue() < numHashMaps, "Hash Map index out of range in hash map slice getHashMapBufferRef!");
    const auto pos = childBufferIndex.getRawValue();
    return hashMapBuffersState[pos] == HashMapBufferState::INITIALIZED ? &hashMapBuffers[pos] : nullptr;
}

const TupleBuffer*
HashMapSlice::getOrCreateHashMapBufferRef(AbstractBufferProvider& bufferProvider, const ChildBufferIndex childBufferIndex)
{
    PRECONDITION(childBufferIndex.getRawValue() < numHashMaps, "Hash Map index out of range in hash map slice loadHashMapBuffer!");
    const auto pos = childBufferIndex.getRawValue();
    /// Check whether the hash map is initialized
    if (hashMapBuffersState[pos] == HashMapBufferState::UNINITIALIZED)
    {
        /// initialize the chained hash map buffer
        /// allocate buffers for the hash maps
        if (auto childBuffer = bufferProvider.getUnpooledBuffer(ChainedHashMap::calculateBufferSizeFromBuckets(params.numberOfBuckets)))
        {
            /// initialize chained hash map i
            ChainedHashMap::init(childBuffer.value(), params.keySize, params.valueSize, params.numberOfBuckets, params.pageSize);
            hashMapBuffers[pos] = childBuffer.value();
            hashMapBuffersState[pos] = HashMapBufferState::INITIALIZED;
        }
        else
        {
            throw BufferAllocationFailure("No unpooled TupleBuffer available for chained hash map child buffer!");
        }
    }
    return &hashMapBuffers[pos];
}

bool HashMapSlice::resetForReuse()
{
    /// State reduction re-reduces a slice once the probe unpins it, so a slice can still be reduced at
    /// garbage collection. Its buffers are gone (serializeState released them), so it cannot be reused.
    if (reduced)
    {
        return false;
    }
    for (uint64_t pos = 0; pos < numHashMaps; ++pos)
    {
        if (hashMapBuffersState[pos] == HashMapBufferState::INITIALIZED)
        {
            ChainedHashMap::load(hashMapBuffers[pos]).clear();
        }
    }
    return true;
}

bool HashMapSlice::matchesLayout(
    const CreateNewHashMapSliceArgs& args, const uint64_t numberOfHashMaps, const uint64_t numberOfInputStreams) const
{
    return numHashmapsPerInputStream == numberOfHashMaps and numInputStreams == numberOfInputStreams and params.keySize == args.keySize
        and params.valueSize == args.valueSize and params.pageSize == args.pageSize
        and ChainedHashMap::calculateBufferSizeFromBuckets(params.numberOfBuckets)
        == ChainedHashMap::calculateBufferSizeFromBuckets(args.numberOfBuckets);
}

uint64_t HashMapSlice::residentBytes() const
{
    return SliceStateCodec::residentBytes(hashMapBuffers);
}

void HashMapSlice::serializeState(std::vector<std::byte>& out)
{
    if (reduced)
    {
        return;
    }
    SliceStateCodec::encodeAndRelease(out, hashMapBuffers);
    /// The slots keep their INITIALIZED marks. A restore puts the same maps back in the same positions,
    /// and marking them UNINITIALIZED would invite getOrCreateHashMapBufferRef to allocate a fresh empty
    /// map over state that is only parked, silently dropping every record in it.
    reduced = true;
}

void HashMapSlice::deserializeState(std::span<const std::byte> in, AbstractBufferProvider& bufferProvider)
{
    if (not reduced)
    {
        return;
    }
    SliceStateCodec::decodeInPlace(in, hashMapBuffers, bufferProvider);

    /// Unlike a PagedVector, a hash map does not survive the move on its own: its chain heads and its
    /// per-entry links are raw pointers into the buffers it had before, which are gone.
    for (uint64_t pos = 0; pos < numHashMaps; ++pos)
    {
        if (hashMapBuffersState[pos] != HashMapBufferState::INITIALIZED)
        {
            continue;
        }
        auto hashMap = ChainedHashMap::load(hashMapBuffers[pos]);
        hashMap.rebuildChains();
    }
    reduced = false;
}
}
