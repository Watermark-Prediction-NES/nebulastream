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
NUMBER_OF_WORKER_THREADS = [1, 8]
OPERATOR_BUFFER_SIZE = [4096, 16384]
### Total worker memory budget in bytes. The worker splits it into a pooled share (operator buffers,
### so pooled buffer count = total * (1 - UNPOOLED_FRACTION) / operator_buffer_size) and an unpooled
### share (var-sized data, hash maps, paged vectors). Exceeding the unpooled share fails the query
### instead of OOM-killing the worker, so it doubles as the pressure the spill policies react to.
TOTAL_MEMORY = [128 * 1024**2, 1024**3]

### Share of TOTAL_MEMORY reserved for unpooled allocations. Not swept — one value for the whole run.
UNPOOLED_FRACTION = 0.5

### Spill sweep. Each dict becomes the SET (...) clause on the query, layered over SPILL_DEFAULTS.
### 'predictor' is consumed only by the predictive and tiered policies; 'compress' and 'backend' are
### orthogonal knobs, so (backend, compress) spans "evict to RAM/disk" x "compressed or not".
SPILL_VARIANTS = [
    {"name": "off", "enabled": False},
    ### Evict without compressing, gated on real memory pressure (high_bound stays at its default).
    {"name": "reactive-mem", "enabled": True, "policy": "reactive", "backend": "in-memory"},
    {"name": "reactive-file", "enabled": True, "policy": "reactive", "backend": "local-file"},
    {"name": "predictive-mem", "enabled": True, "policy": "predictive", "backend": "in-memory"},
    {"name": "predictive-file", "enabled": True, "policy": "predictive", "backend": "local-file"},
    ### Ungated control for the two compress-* rows below: same forced eviction on every GC tick,
    ### compression off. The difference between this and compress-file is compression alone.
    {"name": "always-file", "enabled": True, "policy": "always", "backend": "local-file"},
    ### in-memory + compress == compress without ever touching disk.
    {"name": "compress-mem", "enabled": True, "policy": "always", "backend": "in-memory", "compress": True},
    {"name": "compress-file", "enabled": True, "policy": "always", "backend": "local-file", "compress": True},
    ### One tier per slice from the predicted time-to-trigger, instead of one target for all of them.
    {"name": "tiered-mem", "enabled": True, "policy": "tiered", "backend": "in-memory", "compress": True},
    {"name": "tiered-file", "enabled": True, "policy": "tiered", "backend": "local-file", "compress": True},
    ### Ablation: the same ladder with the compressed rungs switched off, so it degrades to
    ### Resident <-> Disk. Isolates "prediction decides WHEN to evict" from "compression".
    {"name": "tiered-file-nocompress", "enabled": True, "policy": "tiered", "backend": "local-file", "compress": False},
]
PREDICTOR_VARIANTS = ["ewma", "kalman"]

### Filled in for any SPILL_VARIANTS entry that does not name the key. The horizons here are only
### fallbacks: HORIZON_FRACTIONS below overrides them per query (see benchmark.py:horizons_for).
SPILL_DEFAULTS = {
    "compress": False,
    "high_bound": 0.85,
    "horizon_ms": 50,
    "promote_horizon_ms": 20,
    "compress_ram_horizon_ms": 200,
    "compress_disk_horizon_ms": 1000,
}

### Tiered/predictive horizons as a fraction of the window's SLIDE, since slide is what sets the rate
### at which slices become triggerable. Absolute horizons cannot work here: WINDOWS spans 1s to 30s, so
### one fixed ladder either sits entirely inside the promote band for the short windows or entirely
### inside the Disk rung for the long ones. Must be non-decreasing in this order.
HORIZON_FRACTIONS = {
    "promote_horizon_ms": 0.05,
    "horizon_ms": 0.10,
    "compress_ram_horizon_ms": 0.40,
    "compress_disk_horizon_ms": 1.00,
}

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
