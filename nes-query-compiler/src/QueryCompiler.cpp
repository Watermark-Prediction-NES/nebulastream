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


#include <QueryCompiler.hpp>

#include <memory>
#include <Configuration/WorkerConfiguration.hpp>
#include <Phases/LowerToCompiledQueryPlanPhase.hpp>
#include <Phases/LowerToPhysicalOperators.hpp>
#include <Phases/PipeliningPhase.hpp>
#include <Util/DumpMode.hpp>
#include <CompiledQueryPlan.hpp>
#include <ErrorHandling.hpp>

namespace NES::QueryCompilation
{

/// This phase should be as dumb as possible and not further decisions should be made here.
std::unique_ptr<CompiledQueryPlan> QueryCompiler::compileQuery(std::unique_ptr<QueryCompilationRequest> request)
{
    auto lowerToCompiledQueryPlanPhase = LowerToCompiledQueryPlanPhase(request->dumpCompilationResult);
    /// Materialise an effective configuration from the engine defaults. Cheap copy —
    /// QueryExecutionConfiguration is a POD-ish struct.
    auto effectiveConf = defaultQueryExecution;
    effectiveConf.evictionConfiguration = effectiveConf.evictionWorkerConfiguration.toSliceEvictionConfiguration();
    if (request->evictionOverride.has_value())
    {
        /// KNOWN LIMITATION: this REPLACES the whole POD, it does not overlay field by field. The binder
        /// builds the override from SliceEvictionConfiguration's struct defaults (StatementBinder.cpp, the
        /// `SliceEvictionConfiguration cfg{}` in the EVICTION branch) and only writes the keys the query
        /// actually named, so naming ANY eviction key in a SET clause silently resets every worker-level
        /// `eviction.*` setting to its hardcoded default.
        ///
        /// Deliberately not fixed here: a field-by-field merge needs the binder to emit a PARTIAL
        /// override (std::optional per field, or a dedicated SliceEvictionConfigurationOverride type), and the
        /// binder cannot simply seed from the worker config because it lives in nes-sql-parser, one
        /// layer below this one. Until then, a query that sets any EVICTION.* key must set every key it
        /// depends on — see nes-systests/operator/join/JoinEvictionTiered.test.
        effectiveConf.evictionConfiguration = *request->evictionOverride;
    }
    auto queryPlan = LowerToPhysicalOperators::apply(request->queryPlan, effectiveConf);
    auto pipelinedQueryPlan = PipeliningPhase::apply(queryPlan);
    return lowerToCompiledQueryPlanPhase.apply(pipelinedQueryPlan);
}
}
