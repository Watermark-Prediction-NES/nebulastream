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

namespace NES
{

/// Aggregate accuracy metrics over all (target, prediction, truth) triples evaluated.
struct PredictionMetrics
{
    double mae{0.0}; ///< Mean absolute error in wall-clock units.
    double rmse{0.0}; ///< Root mean squared error in wall-clock units.
    double mape{0.0}; ///< Mean absolute percentage error = mean(|predicted - true| / |true|) * 100.
    double maxError{0.0}; ///< Worst-case absolute error in wall-clock units.
    size_t samples{0}; ///< Number of (predicted, true) pairs that contributed.
};

}
