# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build + binary paths for the WatermarkPredictorBenchmark wrapper."""

import os
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
SOURCE_DIR = _BENCH_DIR.parents[2]
### BENCH_BUILD_DIR_OVERRIDE lets you point at an existing build tree (e.g. cmake-build-debug)
### without editing this file. Falls back to cmake-build-bench so the spilling bench cache is reused.
BUILD_DIR = Path(os.environ.get("BENCH_BUILD_DIR_OVERRIDE", SOURCE_DIR / "cmake-build-bench"))
BENCH_BIN = BUILD_DIR / "nes-physical-operators" / "benchmark" / "WatermarkPredictorBenchmark"
RESULTS_ROOT = _BENCH_DIR / "results"

### Build flags. NES_LOG_LEVEL is forced to LEVEL_NONE by the Benchmark build type.
CMAKE_FLAGS = {
    "CMAKE_BUILD_TYPE": "Benchmark",
    "CMAKE_TOOLCHAIN_FILE": "/vcpkg/scripts/buildsystems/vcpkg.cmake",
    "USE_LIBCXX_IF_AVAILABLE": "OFF",
    "ENABLE_LARGE_TESTS": "0",
}
CMAKE_TARGETS = ["WatermarkPredictorBenchmark"]
MOLD_JOBS = 1
