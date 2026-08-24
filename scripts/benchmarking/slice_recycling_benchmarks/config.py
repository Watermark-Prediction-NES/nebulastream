# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All configuration for the slice-recycling / slice-group-creation benchmark sweep."""

import os
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BENCH_DIR.parents[2]
BUILD_DIR = Path(os.environ.get("BENCH_BUILD_DIR_OVERRIDE") or (SOURCE_DIR / "cmake-build-bench"))
SYSTEST_BIN = BUILD_DIR / "nes-systests" / "systest" / "systest"


def testdata_dir() -> Path:
    """With ENABLE_LARGE_TESTS=1 ExternalData stages the downloaded CSVs into the build tree's
    testdata (the source tree only holds .md5 stubs). Resolved lazily: at import time a fresh
    build dir does not contain testdata yet - it appears during the build step."""
    build_testdata = BUILD_DIR / "nes-systests" / "testdata"
    return build_testdata if build_testdata.exists() else SOURCE_DIR / "nes-systests" / "testdata"
RESULTS_ROOT = BENCH_DIR / "results"

### Queries from nes-systests/benchmark_memory/ (Memory source + Void sink). Every query of the
### suite is included; the stateless ones (NM1-NM3, DEBS1-DEBS8) act as a control group where the
### slice knobs must not move the needle.
QUERIES = {
    "CM1": "nes-systests/benchmark_memory/ClusterMonitoring.test:01",
    "CM2": "nes-systests/benchmark_memory/ClusterMonitoring.test:02",
    "LRB1": "nes-systests/benchmark_memory/LinearRoadBenchmark.test:01",
    "LRB2": "nes-systests/benchmark_memory/LinearRoadBenchmark.test:02",
    "MA": "nes-systests/benchmark_memory/Manufacturing.test:01",
    ### 1h/1s sliding window (~3600 slices alive) - heaviest slice-store stress in the suite.
    "SG1": "nes-systests/benchmark_memory/SmartGrid.test:01",
    "SG2": "nes-systests/benchmark_memory/SmartGrid.test:02",
    "YSB1k": "nes-systests/benchmark_memory/YahooStreamingBenchmark.test:01",
    "YSB10k": "nes-systests/benchmark_memory/YahooStreamingBenchmark.test:02",
    "YSBv1": "nes-systests/benchmark_memory/YahooStreamingBenchmark_with_varsized.test:01",
    "YSBv2": "nes-systests/benchmark_memory/YahooStreamingBenchmark_with_varsized.test:02",
    "NM1": "nes-systests/benchmark_memory/Nexmark.test:01",
    "NM2": "nes-systests/benchmark_memory/Nexmark.test:02",
    "NM3": "nes-systests/benchmark_memory/Nexmark.test:03",
    ### The Nexmark join queries need the per-key paged-vector fix in HJBuildPhysicalOperator /
    ### PagedVector::appendPageIfFull (uncommitted, 2026-08-14): upstream sizes every per-key paged
    ### vector's pages at operator_buffer_size (128KiB here) for ~10 records, so the hash-join state
    ### exceeded the 64GiB unpooled budget whenever GC fell ~3% of the stream behind the build
    ### (bimodal ~60-100% failure rate). With the fix: 10/10 clean runs at ~11-17s.
    "NM5": "nes-systests/benchmark_memory/Nexmark.test:04",
    "NM8_Variant": "nes-systests/benchmark_memory/Nexmark.test:05",
    ### The varsized Nexmark inputs are 6GB+ CSVs; the Memory source keeps the whole file in RAM on
    ### top of the query state, so run these only on a big-memory host (>=32GiB for NMv6).
    ### The varsized JOINS (NMv8, NMv8_Variant) fail even with the paged-vector fix: every distinct
    ### key additionally parks its varsized (TEXT) data in its own >=128KiB buffer chain
    ### (PagedVectorRef makeVarSizedAllocFunction), so the per-key state amplification survives one
    ### layer deeper - unpooled budget dies again, and the join output burst can also drain the
    ### pooled share. Engine-level per-key storage design issue; kept in the sweep so the failure
    ### stays visible.
    "NMv1": "nes-systests/benchmark_memory/Nexmark_with_varsized.test:01",
    "NMv2": "nes-systests/benchmark_memory/Nexmark_with_varsized.test:02",
    "NMv3": "nes-systests/benchmark_memory/Nexmark_with_varsized.test:03",
    "NMv5": "nes-systests/benchmark_memory/Nexmark_with_varsized.test:04",
    "NMv8": "nes-systests/benchmark_memory/Nexmark_with_varsized.test:05",
    "NMv8_Variant": "nes-systests/benchmark_memory/Nexmark_with_varsized.test:06",
    "DEBS1": "nes-systests/benchmark_memory/DEBS.test:01",
    "DEBS2": "nes-systests/benchmark_memory/DEBS.test:02",
    "DEBS3": "nes-systests/benchmark_memory/DEBS.test:03",
    "DEBS4": "nes-systests/benchmark_memory/DEBS.test:04",
    "DEBS5": "nes-systests/benchmark_memory/DEBS.test:05",
    "DEBS6": "nes-systests/benchmark_memory/DEBS.test:06",
    "DEBS7": "nes-systests/benchmark_memory/DEBS.test:07",
    "DEBS8": "nes-systests/benchmark_memory/DEBS.test:08",
    "DEBS9": "nes-systests/benchmark_memory/DEBS.test:09",
    "DEBS10": "nes-systests/benchmark_memory/DEBS.test:10",
}

### Sweep axes. The cell list is pruned to where each knob is actually read: max_slice_group_size
### only while group creation is on, slice_pool_capacity only while recycling is on. Per query and
### thread count that leaves 4 cells with recycling off + 8 with it on = 12.
ENABLE_SLICE_RECYCLING = [False, True]
ENABLE_SLICE_GROUP_CREATION = [False, True]
### Measured 2026-08-20 on CM1 at SIZE 100 MS / ADVANCE 10 MS: group creation materialises a slice
### for every slide in [minTs, claimMaxTs] whether or not tuples land there, so slices_created scales
### with this knob - 713,860 (off) -> 2.35M (256) -> 8.65M (1024) -> 22.9M (4096), and e2e with it
### (27.8s -> 34.5s -> 60.1s -> 121.1s at t1; the 4096 x t16 cells time out). slices_wasted never
### exceeded 628, so there were no creation races to win in the first place. The
### spanSlices > 10 * maxSliceGroupSize bail in DefaultTimeBasedSliceStore.cpp:174 is what limits the
### damage, so a LOW cap is the protective setting: 256 keeps the bail engaged on wide buffers.
MAX_SLICE_GROUP_SIZE = [256]
DEFAULT_MAX_SLICE_GROUP_SIZE = 256
WORKER_THREADS = [1, 16]
### Read only when slice recycling is on - DefaultTimeBasedSliceStore gates both the acquire and the
### release on sliceRecyclingEnabled - so the axis is pruned to the recycling-on cells, like
### max_slice_group_size is to group-creation-on.
###
### PER SHARD since the pool was sharded (one shard per worker thread, MaxSlicePoolShards ceiling), so
### the store holds this many times the thread count. That makes 256 the control arm, not a third data
### point: at 16 threads 256 x 16 = 4096 total, i.e. exactly the total the unsharded run held when it
### came out 3% SLOWER than baseline. 256-vs-that isolates the lock removal at equal capacity; 4096
### (65,536 total at t16) then adds the capacity effect on top. At 1 thread there is one shard, so 4096
### reproduces the 1.142x cell directly.
###
### 0 = auto = max(windowSize/windowSlide, 1), which is 1 for a tumbling window. GC retires whatever
### the watermark passed since the last buffer (~2500 slices at TUMBLING 1 MS), so auto pools one per
### shard and destroys the rest.
SLICE_POOL_CAPACITY = [0, 256, 4096]
DEFAULT_SLICE_POOL_CAPACITY = 0

### ThroughputListener interval; 0 disables. Lines land in the per-cell systest log file.
THROUGHPUT_LOG_INTERVAL_MS = 1000

### Worker memory config: total budget is split between the pooled operator buffers and the
### unpooled share (var-sized data, hash maps, paged vectors). 128KiB buffers, not 1MiB: the
### Memory source ingests at RAM speed (no disk throttle), and the window-trigger bursts of the
### sliding-window queries exhaust a coarse-grained pool ("Global buffer pool could not allocate
### buffer before timeout") - 160k x 128KiB absorbs them, 20k x 1MiB does not.
### Totals match the CI large-test budget (nes-systests/systest/CMakeLists.txt LARGE_TEST_*):
### ~71.6GiB total with 64GiB unpooled - the NM5/NM8 join state needs the big unpooled share
### ("No unpooled TupleBuffer available!" with a 20GiB share).
BUFFER_SIZE_BYTES = 131_072
TOTAL_MEMORY = 76_911_476_736
UNPOOLED_FRACTION = 0.8935
PAGE_SIZE = 8192

### Per-query overrides: (operator_buffer_size, total_memory_in_bytes).
BUFFER_OVERRIDES = {}

### The join-heavy Nexmark queries cannot run single-threaded against the memory source: one worker
### cannot probe/GC as fast as the firehose builds state, so state grows until the unpooled budget
### bursts ("No unpooled TupleBuffer available!"). CI also never runs them below 4 threads.
THREAD_OVERRIDES = {
    "NM5": [16],
    "NM8_Variant": [16],
    "NMv5": [16],
    "NMv8": [16],
    "NMv8_Variant": [16],
}

### Per-cell wall-clock cap; on timeout the cell is recorded as failed, not fatal to the sweep.
CELL_TIMEOUT_S = 600

### Runs per cell, averaged. Raise to 3 for final paper numbers.
RUNS_PER_CELL = 3

### NES_LOG_LEVEL=INFO is mandatory: the Benchmark build type otherwise caches LEVEL_NONE and
### compiles out the "Throughput for query" and "Slice creation statistics" lines the harness parses.
### BENCH_VCPKG_TOOLCHAIN overrides the container default for native (non-docker) runs,
### e.g. /home/nschubert/remote_server/vcpkg/scripts/buildsystems/vcpkg.cmake on the tower/amd hosts.
CMAKE_FLAGS = {
    "CMAKE_BUILD_TYPE": "Benchmark",
    "NES_LOG_LEVEL:STRING": "INFO",
    "CMAKE_TOOLCHAIN_FILE": os.environ.get("BENCH_VCPKG_TOOLCHAIN", "/vcpkg/scripts/buildsystems/vcpkg.cmake"),
    "USE_LIBCXX_IF_AVAILABLE": "OFF",
    "ENABLE_LARGE_TESTS": "1",
    "NES_BUILD_NATIVE:BOOL": "ON",
}
### test-data triggers the ExternalData download of the large benchmark CSVs.
CMAKE_TARGETS = ["systest", "test-data"]
MOLD_JOBS = 1
