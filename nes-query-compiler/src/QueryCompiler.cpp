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
    /// Materialise an effective configuration by overlaying the per-query spill override (if any)
    /// onto the engine defaults. Cheap copy — QueryExecutionConfiguration is a POD-ish struct.
    auto effectiveConf = defaultQueryExecution;
    effectiveConf.spillConfiguration = effectiveConf.spillWorkerConfiguration.toSpillConfiguration();
    if (request->spillOverride.has_value())
    {
        /// A `SET (SPILL.* AS ...)` clause owns the policy knobs, but not the deployment-level plumbing:
        /// the spill directory, the I/O thread count and the statistics sink are properties of the
        /// worker, not of a query, and a query that names any SPILL option must not silently reset them.
        const auto& override = *request->spillOverride;
        auto& effective = effectiveConf.spillConfiguration;
        effective.enabled = override.enabled;
        effective.policyName = override.policyName;
        effective.storageBackendName = override.storageBackendName;
        effective.highMemoryBound = override.highMemoryBound;
        effective.predictionHorizon = override.predictionHorizon;
        effective.predictorName = override.predictorName;
    }
    auto queryPlan = LowerToPhysicalOperators::apply(request->queryPlan, effectiveConf);
    auto pipelinedQueryPlan = PipeliningPhase::apply(queryPlan);
    return lowerToCompiledQueryPlanPhase.apply(pipelinedQueryPlan);
}
}
