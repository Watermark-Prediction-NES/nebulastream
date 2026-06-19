# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_source_files(nes-physical-operators
        InMemoryStorageBackend.cpp
        LocalFileStorageBackend.cpp
        NLJSliceStateSerializer.cpp
        HJSliceStateSerializer.cpp
        StubSliceSerializers.cpp
)

add_plugin(REACTIVE SpillPolicy nes-physical-operators ReactiveSpillPolicy.cpp)
add_plugin(PREDICTIVE SpillPolicy nes-physical-operators PredictiveSpillPolicy.cpp)
add_plugin(NEVER SpillPolicy nes-physical-operators NeverSpillPolicy.cpp)
