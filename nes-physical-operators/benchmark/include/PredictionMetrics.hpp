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
#include <cstdint>

namespace NES
{

/// One scored prediction from the rolling (prequential) evaluation: after observing the eval-tick
/// sample, the predictor was asked for the wall-clock crossing of (last watermark + horizon).
/// Aggregation (MAE/RMSE/MAPE/...) is done downstream in the plotting notebook, not here.
struct PredictionSample
{
    size_t evalOffset; ///< Samples since warm-up ended (0 = first online prediction).
    uint64_t horizon; ///< Event-time look-ahead added to the last observed watermark.
    double absErr; ///< |predicted - true| wall-clock.
    double signedErr; ///< predicted - true wall-clock (sign = early/late).
    double trueWall; ///< Ground-truth crossing wall-clock; MAPE denominator (use only when > 0).
};

}
