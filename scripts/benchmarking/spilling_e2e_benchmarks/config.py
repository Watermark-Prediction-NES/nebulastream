# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All benchmark knobs. Edit lists/dicts; benchmark.py picks them up via cartesian product."""

import os
from pathlib import Path

### Paths (derived from this file's location so the same config.py works under
### /tmp/nebulastream/... (local docker) and under
### /home/nschubert/remote_server/watermark-estimation/nebulastream/... (amd7950x3d).
_BENCH_DIR = Path(__file__).resolve().parent
SOURCE_DIR = _BENCH_DIR.parents[2]
### BENCH_BUILD_DIR_OVERRIDE lets you point benchmark.py at a different build tree (e.g.
### cmake-build-debug) for verification without editing this file. Falls back to cmake-build-bench.
BUILD_DIR = Path(os.environ.get("BENCH_BUILD_DIR_OVERRIDE", SOURCE_DIR / "cmake-build-bench"))
WORKER_BIN = BUILD_DIR / "nes-single-node-worker" / "nes-single-node-worker"
NES_CLI_BIN = BUILD_DIR / "nes-frontend" / "apps" / "nes-cli"
GENERATOR_DIR = _BENCH_DIR / "generator"
GENERATOR_BIN = GENERATOR_DIR / "target" / "release" / "spilling-bench-tcp-gen"
RESULTS_ROOT = _BENCH_DIR / "results"
TEMPLATES_DIR = _BENCH_DIR / "templates"

### Build flags. NES_LOG_LEVEL is forced to LEVEL_NONE by the Benchmark build type;
### leave it unset here so the build type owns the value.
CMAKE_FLAGS = {
    "CMAKE_BUILD_TYPE": "Benchmark",
    "CMAKE_TOOLCHAIN_FILE": "/vcpkg/scripts/buildsystems/vcpkg.cmake",
    "USE_LIBCXX_IF_AVAILABLE": "OFF",
    "ENABLE_LARGE_TESTS": "0",
}
CMAKE_TARGETS = ["nes-single-node-worker", "nes-cli"]
MOLD_JOBS = 1

### Worker config sweep. Cartesian product over every list below.
NUMBER_OF_WORKER_THREADS = [1, 4, 8]
OPERATOR_BUFFER_SIZE = [4096, 16384]
NUMBER_OF_BUFFERS = [4096, 32768]

### Spill sweep. Each dict becomes the SET (...) clause on the query.
### 'predictor' is consumed only when policy == 'predictive'.
SPILL_VARIANTS = [
    {"name": "off", "enabled": False},
    {"name": "reactive-mem", "enabled": True, "policy": "reactive", "backend": "in-memory"},
    {"name": "reactive-file", "enabled": True, "policy": "reactive", "backend": "local-file"},
    {"name": "predictive-mem", "enabled": True, "policy": "predictive", "backend": "in-memory"},
    {"name": "predictive-file", "enabled": True, "policy": "predictive", "backend": "local-file"},
]
PREDICTOR_VARIANTS = ["ewma", "kalman"]

### value2 is the VARSIZED column; the generator emits a fixed ASCII payload of this byte count.
VAR_SIZED_BYTES = [16, 256, 4096]

### Ingestion rate sweep: tuples/sec PER SOURCE, passed to the generator's --rate. 0 = unbounded (as fast as the socket accepts).
INGESTION_RATES = [100_000, 1_000_000, 0]

### Tumbling is modelled as SLIDING with size == slide. Tuples are (size_ms, slide_ms).
WINDOWS = [
    (1_000, 1_000),
    (5_000, 5_000),
    (10_000, 2_000),
    (30_000, 5_000),
]

### Runtime knobs.
QUERY_SUBMIT_INTERVAL_S = 3
POST_SUBMIT_DRAIN_S = 30
### Sampling cadence for both the C++ buffer-usage monitor and the Python RSS poller.
MEMORY_SAMPLE_INTERVAL_MS = 100
GENERATOR_PORT_BASE = 9100
GRPC_PORT = 8080
DATA_ADDRESS = "localhost:9090"
WORKER_READY_TIMEOUT_S = 30
GENERATOR_READY_TIMEOUT_S = 10
WORKER_SIGTERM_GRACE_S = 30
GENERATOR_SIGTERM_GRACE_S = 2
