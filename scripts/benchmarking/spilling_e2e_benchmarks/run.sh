#!/usr/bin/env bash

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

### Must be invoked from the nebulastream repo root so paths resolve.
if [[ ! -f "./CMakeLists.txt" || ! -d "./nes-common" || ! -d "./nes-single-node-worker" ]]; then
    echo "error: run.sh must be invoked from the nebulastream repo root." >&2
    echo "       cwd=$(pwd)" >&2
    echo "       expected to find ./CMakeLists.txt + ./nes-common/ + ./nes-single-node-worker/" >&2
    exit 2
fi
if ! grep -qE '^project\s*\(\s*NES\b|^project\s*\(\s*NebulaStream\b' ./CMakeLists.txt; then
    echo "error: ./CMakeLists.txt does not look like the NebulaStream project root." >&2
    exit 2
fi

LOCAL_SOURCE_DIR="$(pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-nebulastream/nes-development:local}"
BENCH_REL="scripts/benchmarking/spilling_e2e_benchmarks"

### Bootstrap payload run inside the container: venv → install → run → trap-cleanup.
### Single-quoted heredoc → no host-side interpolation; $BENCH_DIR, $VENV, $@ resolve
### in the container shell.
read -r -d '' BOOTSTRAP <<'EOS' || true
set -euo pipefail
cd "$BENCH_DIR"

VENV=.venv
trap 'deactivate 2>/dev/null || true; rm -rf "$VENV"' EXIT

echo "[bench] creating venv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

python3 benchmark.py "$@"
EOS

### Ensure :local matches the current deps before benchmarking. Default path rebases :local on the
### matching prebuilt remote image (fast); if none exists, -y mode exits non-zero, so ask before the
### slow local dependency rebuild (-l). libstdcxx + no sanitizer matches the benchmark CMAKE_FLAGS.
if ! scripts/install-local-docker-environment.sh -y --libstdcxx --no-sanitizer; then
    read -r -p "[bench] No matching remote image. Rebuild dependencies locally (-l, slow)? [Y/n] " REBUILD
    case "$REBUILD" in
        [nN]|[nN][oO]) echo "[bench] aborting: no current ${DOCKER_IMAGE} image." >&2; exit 1 ;;
        *) echo "[bench] no matching remote image; building ${DOCKER_IMAGE} locally (-l, slow)..."
           scripts/install-local-docker-environment.sh -y --libstdcxx --no-sanitizer -l ;;
    esac
fi

### -it auto-suppressed when stdin/stdout aren't ttys so the
### same script works under `ssh amd7950x3d "...run.sh ..."` (non-interactive) and from a Mac terminal.
DOCKER_TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
    DOCKER_TTY_FLAGS+=(-it)
fi

### The shared `ccache` named volume is created root-owned, but the dev image runs as a non-root user
### on Linux, so ccache fails with "Permission denied". Chown it to the build user first via a root
### helper container (idempotent). No-op on macOS, where the container already runs as root.
docker run --rm -u 0 -v ccache:/ccache --entrypoint= "${DOCKER_IMAGE}" \
    chown -R "$(id -u):$(id -g)" /ccache

docker run --rm ${DOCKER_TTY_FLAGS[@]+"${DOCKER_TTY_FLAGS[@]}"} \
    -e CCACHE_DIR=/ccache \
    -e BENCH_DIR="/tmp/nebulastream/${BENCH_REL}" \
    -v ccache:/ccache \
    -v "${LOCAL_SOURCE_DIR}:/tmp/nebulastream" \
    --entrypoint= \
    "${DOCKER_IMAGE}" \
    /bin/bash -c "${BOOTSTRAP}" -- "$@"
