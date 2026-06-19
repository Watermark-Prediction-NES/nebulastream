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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <SliceStore/Spill/SliceStateSerializer.hpp>

namespace NES
{

/// Dispatches per Slice subclass via a string name (like SpillPolicyRegistry / StorageBackendRegistry).
/// The handler is told its serializer name at construction (in the lowering rule) and the spilling
/// store looks it up here once. A missing entry means "this Slice subclass does not support spilling".
class SliceStateSerializerRegistry
{
public:
    static SliceStateSerializerRegistry& instance() noexcept
    {
        static SliceStateSerializerRegistry singleton;
        return singleton;
    }

    void registerFor(std::string name, std::shared_ptr<SliceStateSerializer> serializer)
    {
        std::lock_guard guard{mutex};
        serializers.emplace(std::move(name), std::move(serializer));
    }

    [[nodiscard]] SliceStateSerializer* lookup(const std::string& name) const noexcept
    {
        std::lock_guard guard{mutex};
        if (const auto it = serializers.find(name); it != serializers.end())
        {
            return it->second.get();
        }
        return nullptr;
    }

    [[nodiscard]] bool contains(const std::string& name) const noexcept
    {
        std::lock_guard guard{mutex};
        return serializers.contains(name);
    }

private:
    SliceStateSerializerRegistry() = default;

    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<SliceStateSerializer>> serializers;
};

class SliceStateSerializerRegistrar
{
public:
    SliceStateSerializerRegistrar(std::string name, std::shared_ptr<SliceStateSerializer> serializer)
    {
        SliceStateSerializerRegistry::instance().registerFor(std::move(name), std::move(serializer));
    }
};

}
