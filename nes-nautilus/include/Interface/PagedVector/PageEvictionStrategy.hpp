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

#include <cstddef>
#include <vector>

namespace NES
{

class PagedVector;

/// Decides the order in which pages of a PagedVector should be flushed during a spill.
/// Returned indices are strictly increasing; empty == "no preference".
class PageEvictionStrategy
{
public:
    virtual ~PageEvictionStrategy() = default;

    PageEvictionStrategy() = default;
    PageEvictionStrategy(const PageEvictionStrategy&) = delete;
    PageEvictionStrategy(PageEvictionStrategy&&) = delete;
    PageEvictionStrategy& operator=(const PageEvictionStrategy&) = delete;
    PageEvictionStrategy& operator=(PageEvictionStrategy&&) = delete;

    [[nodiscard]] virtual std::vector<std::size_t> orderForEviction(const PagedVector& pv) const = 0;
};

}
