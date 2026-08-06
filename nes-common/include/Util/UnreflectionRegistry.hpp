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

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <Util/ReflectionFwd.hpp>

namespace NES
{

/// The unreflection registry is a standalone counterpart to BaseRegistry that dispatches
/// deserialization by name without compile-time knowledge of the concrete plugin types.
/// Entries are populated by per-plugin glue translation units at static initialization
/// time; the linker keeps those TUs alive via --whole-archive applied to a dedicated glue
/// sub-library (see cmake/UnreflectionRegistrationUtil.cmake).
template <typename ConcreteRegistry, typename KeyTypeT, typename ReturnTypeT>
class UnreflectionRegistry
{
protected:
    UnreflectionRegistry() = default;

public:
    using KeyType = KeyTypeT;
    using ReturnType = ReturnTypeT;
    using UnreflectorFn = std::function<ReturnTypeT(const Reflected&, const ReflectionContext&)>;

    UnreflectionRegistry(const UnreflectionRegistry&) = delete;
    UnreflectionRegistry(UnreflectionRegistry&&) noexcept = delete;
    UnreflectionRegistry& operator=(const UnreflectionRegistry&) = delete;
    UnreflectionRegistry& operator=(UnreflectionRegistry&&) noexcept = delete;
    ~UnreflectionRegistry() = default;

    static ConcreteRegistry& instance()
    {
        static ConcreteRegistry inst;
        return inst;
    }

    /// Register an unreflector under `name`, owned by `owner` -- the C++ type the entry unreflects to.
    /// Returns false only on a genuine collision: `name` is already registered by a *different* owner.
    /// The caller decides how to react; the generated glue throws at static initialization time so a
    /// real conflict is loud and unrecoverable, while a runtime plugin-loading path can roll back or
    /// report it instead of leaving the registry half-initialized.
    ///
    /// Re-registering the same (name, owner) is accepted and keeps the first entry. That is not
    /// laxity. The generated glue translation units have internal linkage and live in a sub-library
    /// that reaches consumers through an INTERFACE link line wrapped in WHOLE_ARCHIVE, so a consumer
    /// that reaches the parent target through more than one path in the static-library graph links
    /// several copies of the same object -- with no duplicate-symbol error, precisely because the
    /// symbols are internal. Each copy then runs its own initializer. Treating that as a conflict
    /// aborts the process before main() and takes every test binary with it.
    ///
    /// `name` is taken by reference so the caller can still inspect it on the failure path regardless
    /// of how the underlying map handles its arguments.
    [[nodiscard]] bool addUnreflectorEntry(const KeyTypeT& name, const std::string_view owner, UnreflectorFn unreflectorFunction)
    {
        const auto [entry, inserted] = unreflectorTable.try_emplace(name, Entry{std::move(unreflectorFunction), std::string{owner}});
        return inserted or entry->second.owner == owner;
    }

    [[nodiscard]] bool contains(const KeyTypeT& name) const { return unreflectorTable.contains(name); }

    [[nodiscard]] std::optional<ReturnTypeT> unreflect(const KeyTypeT& name, const Reflected& data, const ReflectionContext& context) const
    {
        if (const auto it = unreflectorTable.find(name); it != unreflectorTable.end())
        {
            return it->second.unreflector(data, context);
        }
        return std::nullopt;
    }

private:
    /// The owner is kept so a repeated registration can be told apart from a genuine key collision.
    struct Entry
    {
        UnreflectorFn unreflector;
        std::string owner;
    };

    std::unordered_map<KeyTypeT, Entry> unreflectorTable;

    friend ConcreteRegistry;
};

}
