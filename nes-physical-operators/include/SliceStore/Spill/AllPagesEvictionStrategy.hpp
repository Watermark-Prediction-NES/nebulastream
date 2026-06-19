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
#include <Interface/PagedVector/PageEvictionStrategy.hpp>
#include <Interface/PagedVector/PagedVector.hpp>

namespace NES
{

/// Default eviction strategy: evict all pages, in their natural (append) order.
/// Sufficient for v1 since NLJ and aggregation slices spill the full slice at once.
class AllPagesEvictionStrategy final : public PageEvictionStrategy
{
public:
    [[nodiscard]] std::vector<std::size_t> orderForEviction(const PagedVector& pv) const override
    {
        std::vector<std::size_t> indices;
        indices.reserve(pv.getNumberOfPages());
        for (std::size_t i = 0; i < pv.getNumberOfPages(); ++i)
        {
            indices.push_back(i);
        }
        return indices;
    }
};

}
