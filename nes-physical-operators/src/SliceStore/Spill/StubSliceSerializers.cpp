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

#include <cstdint>
#include <expected>
#include <future>
#include <memory>
#include <utility>
#include <Runtime/AbstractBufferProvider.hpp>
#include <SliceStore/Slice.hpp>
#include <SliceStore/Spill/SliceStateSerializer.hpp>
#include <SliceStore/Spill/SliceStateSerializerRegistry.hpp>
#include <SliceStore/Spill/StorageBackend.hpp>

namespace NES
{

namespace
{
template <typename T>
std::future<T> ready(T value)
{
    std::promise<T> promise;
    auto future = promise.get_future();
    promise.set_value(std::move(value));
    return future;
}

/// Registered placeholder used until aggregation-slice spill is implemented.
/// The decorator's GC tick will lookup() this stub, see spill() return a "not yet implemented" error, log a WARN,
/// and skip the slice. No data loss, no crash — queries that opt in to spill simply observe that
/// these slice types are not yet spillable. AggregationSlice (HashMap state) is non-trivial and
/// deferred to a follow-up PR; HJSlice is now handled by HJSliceStateSerializer.
class NotImplementedSliceStateSerializer final : public SliceStateSerializer
{
public:
    explicit NotImplementedSliceStateSerializer(const char* name) noexcept : name(name) { }

    [[nodiscard]] std::future<std::expected<SpilledSliceHandle, IoError>>
    spill(Slice& /*slice*/, StorageBackend& /*backend*/, AbstractBufferProvider& /*buffers*/) override
    {
        return ready<std::expected<SpilledSliceHandle, IoError>>(
            std::unexpected{IoError{IoErrorCode::TransientIo, std::string{name} + " spill not yet implemented"}});
    }

    [[nodiscard]] std::future<std::expected<void, IoError>> restore(
        Slice& /*slice*/, const SpilledSliceHandle& /*handle*/, StorageBackend& /*backend*/, AbstractBufferProvider& /*buffers*/) override
    {
        return ready<std::expected<void, IoError>>(
            std::unexpected{IoError{IoErrorCode::TransientIo, std::string{name} + " restore not yet implemented"}});
    }

    [[nodiscard]] uint64_t residentBytes(const Slice& /*slice*/) const noexcept override { return 0; }

private:
    const char* name;
};

const SliceStateSerializerRegistrar aggRegistrar{
    "AggregationSlice", std::make_shared<NotImplementedSliceStateSerializer>("AggregationSlice")};
}

}
