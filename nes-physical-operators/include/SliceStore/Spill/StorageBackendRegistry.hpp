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

#include <cctype>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <SliceStore/Spill/StorageBackend.hpp>

namespace NES
{

/// Arguments for constructing a StorageBackend from the registry.
struct StorageBackendArgs
{
    std::string spillDirectory{"/tmp/nes-spill"};
    uint32_t ioThreads{4};
};

class StorageBackendRegistry
{
public:
    using Factory = std::function<std::shared_ptr<StorageBackend>(const StorageBackendArgs&)>;

    static StorageBackendRegistry& instance() noexcept
    {
        static StorageBackendRegistry singleton;
        return singleton;
    }

    void registerFactory(std::string name, Factory factory)
    {
        std::lock_guard guard{mutex};
        factories.emplace(toUpper(std::move(name)), std::move(factory));
    }

    [[nodiscard]] bool contains(std::string_view name) const
    {
        std::lock_guard guard{mutex};
        return factories.contains(toUpper(std::string{name}));
    }

    [[nodiscard]] std::shared_ptr<StorageBackend> create(std::string_view name, const StorageBackendArgs& args) const
    {
        std::lock_guard guard{mutex};
        if (const auto it = factories.find(toUpper(std::string{name})); it != factories.end())
        {
            return it->second(args);
        }
        return nullptr;
    }

    [[nodiscard]] std::vector<std::string> registeredNames() const
    {
        std::lock_guard guard{mutex};
        std::vector<std::string> names;
        names.reserve(factories.size());
        for (const auto& [name, _] : factories)
        {
            names.push_back(name);
        }
        return names;
    }

private:
    StorageBackendRegistry() = default;

    static std::string toUpper(std::string s)
    {
        for (auto& c : s)
        {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        return s;
    }

    mutable std::mutex mutex;
    std::unordered_map<std::string, Factory> factories;
};

class StorageBackendRegistrar
{
public:
    StorageBackendRegistrar(std::string name, StorageBackendRegistry::Factory factory)
    {
        StorageBackendRegistry::instance().registerFactory(std::move(name), std::move(factory));
    }
};

}
