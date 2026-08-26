#!/usr/bin/env bats

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Guards the docker identity every distributed suite is given. nes_cleanup_leaked_resources removes
# containers by network label and images by tag prefix, so two suites sharing either one tear down
# each other's resources mid-run. Three suites drive nes-cli and used to share both, which failed as
# "No such image" or an unhealthy container, but only when their runs happened to overlap.
# No docker daemon is needed: the derivation is pure and the rest is grep.

source "$NES_BATS_LIB"

setup_file() {
  nes_require_env NES_DIR
}

# The path a suite passes is only ever read with basename, so a stand-in is enough here and keeps
# the test independent of whether the binaries have been built.
bin_path_for() {
  local var="$1"
  if [ -n "${!var:-}" ]; then
    echo "${!var}"
  else
    echo "/nonexistent/$(echo "$var" | tr 'A-Z_' 'a-z-')"
  fi
}

# Source copies only. A build tree stages these files (test-tmp, copied testdata), and counting a
# staged copy as a second suite would make every run after a docker suite look like a collision.
# The remote verify hosts are not git checkouts, hence the find fallback.
source_files() {
  local pattern="$1"
  if git -C "$NES_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$NES_DIR" ls-files -- "$pattern" | sed "s|^|$NES_DIR/|"
  else
    find "$NES_DIR" -name "$pattern" \
      -not -path '*/cmake-build*' -not -path '*/build/*' -not -path '*/test-tmp/*' \
      -not -path '*/_deps/*' -not -path '*/vcpkg_installed/*'
  fi
}

# One "<binary var> <suite>" line per suite that registers itself with the distributed preset.
registered_suites() {
  local f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    sed -nE 's/.*setup_file +"\$([A-Z_]+)"[[:space:]]*([a-z0-9-]*).*/\1 \2/p' "$f" | head -1
  done < <(source_files '*.bats')
}

@test "each nes-cli suite gets its own label and image prefixes" {
  nes_derive_image_names /usr/bin/nes-cli cli
  local cli_label="$NES_BATS_TEST_LABEL" cli_worker="$NES_BATS_WORKER_PREFIX" cli_app="$NES_BATS_APP_PREFIX"

  nes_derive_image_names /usr/bin/nes-cli mqtt-source
  [ "$NES_BATS_TEST_LABEL" != "$cli_label" ]
  [ "$NES_BATS_WORKER_PREFIX" != "$cli_worker" ]
  [ "$NES_BATS_APP_PREFIX" != "$cli_app" ]

  # The compose files look this one up by name, so it stays keyed on the binary.
  [ "$NES_BATS_APP_IMAGE_VAR" = "CLI_IMAGE" ]
}

@test "suite name defaults to the binary when only one suite drives it" {
  nes_derive_image_names /usr/bin/nes-repl
  [ "$NES_BATS_TEST_LABEL" = "distributed-repl" ]
  [ "$NES_BATS_WORKER_PREFIX" = "nes-worker-repl-test" ]
  [ "$NES_BATS_APP_IMAGE_VAR" = "REPL_IMAGE" ]
}

@test "no two registered suites share a docker identity" {
  local labels="" workers="" apps="" var suite

  run registered_suites
  [ "$status" -eq 0 ]
  [ -n "$output" ]

  while read -r var suite; do
    [ -n "$var" ] || continue
    nes_derive_image_names "$(bin_path_for "$var")" "$suite"
    labels+="$NES_BATS_TEST_LABEL"$'\n'
    workers+="$NES_BATS_WORKER_PREFIX"$'\n'
    apps+="$NES_BATS_APP_PREFIX"$'\n'
  done <<< "$output"

  local dupes
  for names in "$labels" "$workers" "$apps"; do
    dupes=$(printf '%s' "$names" | sort | uniq -d)
    [ -z "$dupes" ] || {
      echo "suites share a docker identity, so their cleanup will collide: $dupes" >&2
      return 1
    }
  done
}

@test "no two compose files hardcode the same network label" {
  local dupes
  dupes=$(while IFS= read -r f; do
            [ -n "$f" ] || continue
            sed -nE 's/^ *nes-test: (.*)$/\1/p' "$f"
          done < <(source_files 'create_compose.sh') \
    | sed -E 's/^\$\{NES_BATS_TEST_LABEL:-(.*)\}$/\1/' \
    | sort | uniq -d)
  [ -z "$dupes" ] || {
    echo "compose files share a network label: $dupes" >&2
    return 1
  }
}
