# Exagium: architectural synthesis and research direction

**Status:** architectural proposal and synthesis of discussions through 2026-11-25  
**Scope:** long-range direction, architectural requirements, current implementation gaps, and research questions  
**Decision status:** proposed; not an implementation specification and not an authorisation for autonomous changes

## 1. Executive position

Exagium's long-range ambition is best described as an **evolving, heterogeneous computational discovery environment**.

Its purpose is not merely to run agents, orchestrate existing tools, benchmark models, or automate machine-learning experiments. Those are useful capabilities, but they are means rather than the full objective.

The larger objective is to automate the discovery, creation, composition, evaluation and optimisation of increasingly capable skills, systems, models, algorithms, agents and computational arrangements — eventually including improvement of the processes by which those capabilities are discovered and developed.

The environment should be able to:

1. Investigate a problem, opportunity, limitation or hypothesis.
2. Discover relevant knowledge, capabilities, resources and methods.
3. Decompose work into independently managed, nested investigations.
4. Construct candidate solutions and compositions.
5. Execute them across appropriate computational substrates.
6. Measure behaviour, cost, correctness, uncertainty and limitations.
7. Independently evaluate evidence and compare alternatives.
8. Preserve reusable discoveries, negative results and dependencies.
9. Use those discoveries to improve subsequent investigations and the machinery that conducts them.

This is an architectural direction, not a claim that the current repository already provides all these capabilities.

## 2. Three non-negotiable properties

The architecture should balance three mutually reinforcing requirements.

### 2.1 Capability growth

The system must be able to do more than invoke a fixed catalogue of tools. It should discover, construct, combine and retain capabilities that were not anticipated in its initial design.

### 2.2 Epistemic integrity

The system must distinguish proposals, hypotheses, observations, measurements, independently checked results and validated capabilities. Confidence must not increase merely because an assertion is repeated, composed, or consumed by another agent.

### 2.3 Computational freedom

The system must not be structurally trapped by one model provider, agent harness, programming language, processor, execution model, benchmark, or infrastructure provider. Abstraction must preserve meaningful differences in semantics, precision, determinism, security, resource requirements and cost; it must not flatten unlike systems into a misleading lowest common denominator.

Capability growth without epistemic integrity risks amplifying errors. Epistemic integrity without capability growth produces a system that checks simple work very carefully but cannot expand its abilities. Computational freedom without either creates a large collection of disconnected integrations.

## 3. Operating model: five cooperating engines

The following engines describe responsibilities, not five isolated applications or a mandatory linear pipeline.

### 3.1 Discovery

Research, knowledge acquisition, hypothesis generation, opportunity identification, capability and resource discovery, decomposition, and identification of missing knowledge.

### 3.2 Experimentation and measurement

Candidate execution, simulation, testing, benchmarking, measurement, comparison, uncertainty reduction, failure investigation and independent evaluation.

### 3.3 Composition and construction

Construction of algorithms, software, agents, models, workflows, computational arrangements and larger systems from existing capabilities or newly created components.

### 3.4 Knowledge and evidence

Provenance, claims, observations, results, failures, dependencies, uncertainty, reproducibility, capability descriptions, evidence maturity and retained discoveries.

### 3.5 Meta-optimisation

Improvement of discovery, decomposition, experimentation, measurement allocation, composition, orchestration, computational efficiency, resource utilisation and the processes that improve those processes.

The engines must be able to interrupt, redirect and inform one another. Experimentation may expose a missing capability; discovery may find it; composition may construct it; evidence may qualify or reject it; meta-optimisation may improve the method used to find it. A finding may also invalidate the original objective or reveal a better one.

## 4. Three architectural relationships, not a fixed three-layer ceiling

A useful starting distinction is between three kinds of relationship:

- **World relationship:** a system acts on or investigates an environment external to itself.
- **System relationship:** a system observes, coordinates, allocates resources to, evaluates or changes another system.
- **Self relationship:** a system observes or modifies its own structure, methods and behaviour, including the mechanisms through which it changes other systems.

These relationships can be illustrated as operational, supervisory and evolutionary modes. They should not be treated as three permanently fixed tiers, nor should the third be treated as an omniscient authority outside the system.

A worker can participate in an operational process, expose evidence to a supervisory process, and contribute to an evolutionary investigation. A nested investigation can itself contain operational, supervisory and self-referential responsibilities. Different scales may instantiate similar organisational rules without having identical implementations.

This suggests a **multi-scale, multi-layer system** in which components can participate in several relationship types and cross-layer interactions are explicit.

## 5. Recursive funnel: divergence, construction, convergence and expansion

The recursive funnel is a central operating principle. It must support both divergence and convergence.

A typical investigation can:

1. Receive an objective, opportunity, constraint or hypothesis.
2. Diverge through research, alternatives, decomposition and candidate approaches.
3. Construct experiments, implementations or compositions.
4. Execute and measure them.
5. Converge through comparison, verification, selection, rejection or a finding of insufficient evidence.
6. Preserve results and update the capability and knowledge maps.
7. Generate new opportunities and investigations from what was learned.

This process can occur at multiple scales: a complete project, a research question, a mathematical problem, an algorithm, a hardware configuration or a single experiment.

A parent investigation may contain many child investigations. Children may produce independent reusable capabilities, invalidate assumptions, reveal dependencies, or establish infeasibility. A child completing does not imply its parent objective is complete.

Do not implement this as one giant recursive agent. Use explicit, independently managed investigation objects with bounded authority, budgets, dependencies, lifecycle state and evidence.

### 5.1 Investigation lifecycle must be explicit

At minimum, distinguish:

- Work proposed/generated.
- Work accepted.
- Work sufficiently specified.
- Work executable/claimable.
- Work claimed and running.
- Execution completed.
- Result persisted.
- Result evaluated.
- Evidence accepted or rejected.
- Capability validated or retained.
- Parent objective achieved, redirected, suspended or terminated.

The word "done" must not conflate execution, evidence maturity, hypothesis resolution, capability validation and objective completion.

A thousand generated children are not a thousand units of progress. Progress must be evaluated against the parent objective and the quality of resulting evidence or capabilities.

## 6. Shared capability model

A central missing architectural concept is a common representation of anything Exagium may discover, use, construct, compose or improve.

A capability may be a model, algorithm, mathematical method, software package, hardware accelerator, research technique, data transformation, experiment, workflow, agent, or a complete composed system.

A capability description should include, as applicable:

- Identity, version and provenance.
- Purpose and demonstrated behaviour.
- Inputs, outputs and interface contracts.
- Preconditions, postconditions and operating conditions.
- Supported operations and computational substrates.
- Dependencies and composition relationships.
- Precision, determinism and reproducibility properties.
- Resource, memory, communication and energy requirements.
- Cost, latency, throughput and reliability measurements.
- Security boundaries and authority requirements.
- Evidence maturity, uncertainty, known limitations and failure modes.
- Validity conditions and revalidation requirements.

A provider's description or a model's self-report is not proof of a capability. Separate declared capability from experimentally demonstrated capability and independently validated capability.

Compositions must themselves be representable as capabilities, with explicit dependencies and evidence. This permits a validated composition to become a component of a more complex construction.

## 7. Heterogeneous computational substrate

Exagium must not assume that computation means dispatching a Python script to a CPU worker.

It should be able to reason about:

- **Processing hardware:** CPU, GPU, NPU, FPGA, ASIC and specialised accelerators.
- **Computational models:** sequential, parallel, vector, tensor, graph, symbolic and probabilistic.
- **Execution patterns:** dataflow, streaming, event-driven, distributed and batch.
- **Memory:** caches, RAM, VRAM, unified memory and near-memory arrangements.
- **Communication:** PCIe, networks, interconnects, shared state and data movement.
- **Software systems:** compilers, databases, numerical libraries, simulation engines and runtimes.
- **Emerging substrates:** photonic, analogue and neuromorphic computation where practical evidence and access exist.

Workloads should express requirements and constraints. Implementations and resources should declare what they support. Adapters should preserve meaningful differences rather than imply false equivalence.

A CPU and GPU implementation may produce superficially similar outputs while differing in precision, determinism, memory use, transfer overhead, failure modes and correctness. These differences must be visible to discovery, scheduling and evaluation.

The long-term ambition includes investigating computational arrangements and systems themselves, not only using currently available compute.

## 8. Measurement is a resource-allocation and epistemic problem

Benchmarking is only one part of measurement. The system must account for the full cost of discovery and validation:

- Candidate construction.
- Search and proposal generation.
- Execution.
- Measurement.
- Correctness verification.
- Behavioural understanding.
- Characterisation of limitations.
- Selection, integration and maintenance.

A candidate can be cheap to execute but expensive to discover. A candidate can appear superior because its measurement process is flawed. An experiment can be less valuable than a different measurement that reduces uncertainty more.

Measurement allocation should therefore consider:

- Expected information gain and uncertainty reduction.
- Expected improvement and probability of success.
- Cost of the next experiment.
- Whether the benchmark represents the actual objective.
- Whether candidates are compared under equivalent conditions.
- Whether selection effects or repeated tuning have contaminated evaluation.
- Whether a more informative experiment can eliminate many candidates cheaply.

A key composition test is whether a composition outperforms its strongest individual constituent after accounting for the full cost of composition. See [Issue #103](https://github.com/5kasgard/exagium/issues/103). More agents or components do not inherently imply greater capability or throughput.

## 9. Evidence integrity, error containment and governance

Capability recursion and epistemic recursion are different:

- **Capability recursion** improves what the system can do.
- **Epistemic recursion** improves how reliably it determines what is true.

If capability development outpaces epistemic reliability, the system becomes more powerful without becoming proportionally more trustworthy.

Evidence integrity must therefore surround the entire system, not exist only in a single evidence service.

Required properties include:

- **Provenance:** preserve the origin of consequential claims, inputs, measurements and results.
- **Evidence maturity:** distinguish hypotheses, observations, tested results and validated capabilities.
- **Uncertainty preservation:** composition must not manufacture confidence.
- **Source independence:** repeated use of one source must not count as independent verification.
- **Dependency awareness:** identify downstream processes relying on questionable evidence.
- **Containment:** limit the influence of unverified results.
- **Reversibility:** preserve recoverable prior states and rollback paths.
- **Revalidation:** reassess evidence when conditions, implementations or dependencies change.
- **Separation of duties:** proposers, executors, meters, verifiers, selectors and approvers must not silently collapse into one authority.
- **Bounded authority:** the ability to propose a change must not imply permission to approve, execute and certify it.
- **External governance:** autonomous self-modification must not silently redefine the invariants and approval boundaries that govern it.

A perfectly replayable history can faithfully reconstruct a bad decision. Event completeness, event correctness, causal correctness and evidential validity are separate properties.

## 10. Mathematical architecture and system geometry: research hypothesis

A recent discussion raised a potentially important question: could number theory, geometry and other mathematical structures expose a more fundamental architecture for layering and connecting Exagium?

The hypothesis is that if constraints, relationships and transformations map consistently across scales, the same underlying mathematical structure may describe an operational system, a system managing it, and a system improving both — despite their different functions and implementations.

This is a research direction, not an established design choice. Do not assign arbitrary numbers to layers or impose an attractive sequence without evidence that it provides useful properties.

### 10.1 Candidate mathematical structures

| Mathematical area | Possible use to investigate | Important limitation |
|---|---|---|
| Fibonacci sequences and growth models | Adaptive branching, expansion schedules and resource allocation | No reason to assume Fibonacci growth is optimal or universally natural |
| Prime numbers and factorisation | Decomposition, independent factors and irreducibility under a defined operation | Mathematical primality does not automatically imply software or system irreducibility |
| Modular arithmetic and coprime periods | Scheduling, coordination, partitioning and avoiding synchronisation collisions | Useful periods depend on actual workload and resource behaviour |
| Fractal geometry and self-similarity | Reusable organisational patterns across investigation scales | Similarity of shape does not guarantee equivalent semantics or safety |
| Graphs and hypergraphs | Dependencies, capabilities, compositions and multi-component relationships | Ordinary edges may be insufficient for relationships involving several components together |
| Category theory | Structure-preserving mappings, composition and relationships between different system domains | Abstract elegance does not itself establish an implementable or useful architecture |
| Algebraic structures | Composition rules, identities, inverses, closure and invariants | The relevant operations and assumptions must be defined |
| Dynamical systems and control theory | Feedback, stability, adaptation, timescale separation and system interaction | Stability of a model does not guarantee validity of its model of reality |
| Information theory | Information gain, uncertainty and measurement allocation | Information gain must be tied to useful objectives and trustworthy observations |
| Formal logic and type theory | Contracts, invariants, permissions and correctness properties | A proof establishes properties of a specification, not automatically the real-world adequacy of that specification |

### 10.2 Structural preservation across layers

Investigate distinctions such as:

- **Isomorphism:** structures are equivalent with respect to specified relationships.
- **Homomorphism:** a mapping preserves specified operations or relationships.
- **Invariant:** a property remains unchanged under a transformation.
- **Self-similarity:** a structure resembles itself across scales.
- **Recursion:** a process or structure is defined partly in terms of itself.

Identical-looking constraints do not automatically imply identical behaviour. The research must identify which operations, relations and invariants are preserved by a mapping between scales.

One possible abstract model is a family of systems with a common structural schema, where each instance has different contents, scale, authority and operating conditions. The goal is not necessarily three layers; it is to discover whether a reusable mathematical structure generates valid relationships across many layers and scales.

### 10.3 Proposed comparative research

Compare at least four candidate architectural geometries:

1. Hierarchical/nested systems.
2. Fractal or recursively self-similar structures.
3. Multiplex or multilayer networks with explicit cross-layer relationships.
4. Dynamical/control-system models with evolving state and feedback.

Evaluate each against the same requirements:

- Nested investigation representation.
- Heterogeneous computational resources.
- Capability composition and dependency representation.
- Evidence and uncertainty propagation.
- Failure containment and recovery.
- Controlled self-modification.
- Structural adaptation.
- Coordination overhead and operational complexity.

A hybrid may be more appropriate than any single model: for example, hierarchical governance, recursively structured investigations, multiplex capability relationships and dynamical resource allocation.

### 10.4 Self-reference and regress

If one system manages another and a third manages the manager, the question of what manages the third can recurse indefinitely. Do not solve this by declaring the highest layer omniscient. Investigate bounded authority, protected invariants, independent evaluation, reversibility, external governance and explicit termination conditions.

The system's architecture should be capable of evolving without being able to casually redefine the conditions under which its evolution is judged successful.

## 11. Lean 4 and formal methods

Lean 4 is a possible formal-verification component, not Exagium's organising principle.

Prioritise formalisation of consequential invariants, such as:

- Valid task-state transitions.
- Budget and resource invariants.
- Lease ownership and expiry.
- Parent-child completion requirements.
- Evidence-promotion conditions.
- Permission and authority boundaries.
- Rules governing consequential changes.

Connect specifications and proofs to actual Rust and Python implementations through generated tests where appropriate, property-based testing, runtime assertions, trace comparison and CI enforcement.

A proof about a specification does not prove that an implementation conforms to it, nor that the specification captures every relevant real-world concern. Do not attempt to prove the entire system correct before identifying the highest-consequence invariants.

## 12. Resource ecosystem

The API and infrastructure ecosystem should be treated as a discoverable resource environment, not merely a list of chatbot integrations.

Potential resource categories include:

- Intelligence and inference providers.
- Evaluation and model-analysis services.
- Research indexes and scholarly sources.
- Datasets and data services.
- Sandboxed execution.
- Compute and infrastructure providers.
- Coordination, caching and storage.

Each resource should have a capability description including actual supported operations, constraints, cost, reliability, authentication, privacy and security boundaries, demonstrated performance and failure behaviour.

Providers are not interchangeable merely because they expose superficially similar endpoints. Research sources may expose metadata but not full text; compute providers may not guarantee a particular accelerator; a sandbox may not meet every threat model; model endpoints may differ in tool calling, context, latency and reliability.

The current provider list must not become a permanent boundary. Resource discovery should be extensible.

## 13. Interface and operating environment

The interface is a window into the system, not the system itself.

A replaceable frontend — potentially including the current chat experience or a LibreChat-style interface — should expose shared backend state rather than maintain an independent understanding of ongoing work.

The longer-term workspace may include:

- Conversational interaction.
- Live investigations, queue and process views.
- Capability and dependency graphs.
- Resource and provider explorer.
- Experiment, measurement and evidence explorer.
- Files, code and terminal workspace.
- Nested investigation visualisation.
- Approval and governance controls.
- System health, cost and performance monitoring.

Work must continue independently of a chat session. A user should be able to reconnect from another device and inspect the same running processes, results and decisions. Provider independence should not depend on a paid agent subscription.

## 14. Current repository reality and known gaps

The current README describes a lab where agents build, test, optimise and distil neural networks and other AI against real hardware. Its measured machinery, candidate loop, hardware probes, ledger and router are real foundations. The broader discovery-environment ambition in this document is a proposed direction and must not be confused with already demonstrated repository capability.

The following findings are dated snapshots from the November 24–25, 2026 reviews and repository artefacts. They must be rechecked against current branch heads before being treated as current runtime measurements.

### 14.1 Queue and worker mismatch

The November 24 audit snapshot reported 313 queue items: 127 done, 110 active and 76 parked. It also reported 53 items labelled ready but zero executable open jobs under the worker's actual rules. A separate issue snapshot on November 25 reported 75 open items with zero claimable by a free worker because required job metadata was absent. These are different snapshots and must not be combined as one measurement.

Issue [#150](https://github.com/5kasgard/exagium/issues/150) records the later splitter/claimability mismatch: 72 generated subitems, 52 parked and 8 completed; 41 open items had not been through the splitter. The issue is evidence of a task representation and coordination failure, not merely insufficient model capacity.

The earlier audit also observed poor child conversion (4 of 56 code children done, 46 parked) and 94 of 100 observed worker runs idle. These figures are snapshot-specific and require remeasurement.

### 14.2 Architectural wiring gaps

The November 24 audit identified disconnected or incomplete areas around P9, P11, P12, P17, P20, P21, P22 and P23, including process management, knowledge storage, composition, capability mapping, orchestration, graph representation, allocation and statistical controls.

The key question is not whether these labels or modules exist, but whether their state, data contracts, events and decisions are connected end-to-end and contribute to completed, evidence-backed outcomes.

### 14.3 Other systemic risks identified

The earlier audit identified:

- Experiments capable of self-certifying measurements or verdicts.
- Inconsistent sandbox boundaries between execution paths.
- Non-atomic result persistence and queue completion.
- Overloaded completion semantics.
- Metrics vulnerable to rewarding task fragmentation.
- Microtask limits obstructing coordinated changes.
- No shared capability, cost and evidence model across engines.

These require current-code revalidation before any fix is proposed.

### 14.4 Event replay

The P37 replay report at [audit/spine/RESULT-2026-11-25.md](https://github.com/5kasgard/exagium/blob/pilot/lgn-20261115/audit/spine/RESULT-2026-11-25.md) recorded an initial replay of 230/339 items reconstructed exactly, followed by a P37b update reporting 340/340 reconstructed and 814/814 post-transition non-claim status changes carrying a run with an outcome.

This is meaningful progress in reconstructability. It does not by itself establish that the actions were valid, the outcomes truthful, or the decisions causally justified.

### 14.5 Fabric, chat and connector direction

The newer Fabric, chat, connector and execution work creates a more coherent environment for interaction and shared state. The architectural direction can be understood as:

- Interaction layer: chat, OpenCode and external clients.
- Fabric/connector: shared-state access, inspection, provenance and context.
- Forge/execution: queue, workers, experiments, research, measurements, findings and ledger.

The connector was observed as read-only in the inspected implementation. Inspection is not the same as governed operation or improvement. Write-capable tools require explicit allowlists, argument validation, permission checks, audit records, idempotency and approval rules.

The inspected chat worker also raised implementation questions requiring follow-up: Access JWT claim validation and error handling; MCP HTTP/JSON-RPC error checking; tool allowlisting and schema validation; malformed argument handling; request and tool-result size limits; and bounded tool loops. These are code-review observations, not proof of an exploitable vulnerability. Recheck current code before remediation.

A further version-skew concern was observed: main-branch worker code referenced queue data on a separate pilot branch. This may be intentional, but schema and code compatibility must be explicit and observable before operations are allowed.

## 15. Revised implementation programme

The work should be dependency-aware, not a speculative rewrite.

1. **Establish one coherent system of record.** Reconcile Fabric, queue state, worker state, run records, results and evidence. Define schemas, versions, identities, provenance and consistency rules.
2. **Repair execution and investigation lifecycle.** Fix readiness, child reconciliation, scheduling, leases, result persistence, resource accounting, worker utilisation and parent completion semantics.
3. **Establish capability and evidence models.** Unify descriptions of resources, capabilities, compositions, dependencies, cost, uncertainty and evidence maturity.
4. **Connect discovery and experimentation.** Make research, decomposition, candidate generation, execution, measurement allocation and independent evaluation work as one evidence-backed process.
5. **Make composition a real system capability.** Construct and evaluate combinations, including comparisons against strongest constituents and full composition costs.
6. **Expand heterogeneous computation.** Add workload requirements, hardware and memory awareness, execution adapters, data movement and distributed resources.
7. **Formalise critical invariants and governance.** Use Lean 4 selectively with implementation-level tests, authority boundaries, approval and rollback.
8. **Enable controlled meta-optimisation.** Let Exagium investigate and improve its own discovery, construction, measurement, orchestration and computational processes under protected constraints.
9. **Expose the environment through a unified interface.** Build views over the same state and processes used by the backend.

Some streams can proceed in parallel, but shared contracts and dependencies must be established first. Dynamic composition without shared evidence semantics risks multiplying unreliable results. Autonomous write access without authority and rollback rules risks turning an integration layer into an uncontrolled control plane.

## 16. Evaluation criteria for architectural changes

Every proposed feature, PR or subsystem should answer:

1. Which discovery, construction, composition, evaluation or optimisation capability does it improve?
2. What existing system state and contracts does it depend on?
3. Which other components consume its outputs?
4. Does it increase completed, useful outcomes rather than generated work?
5. What evidence demonstrates its contribution?
6. What is the full cost, including search, construction, execution, measurement, verification and maintenance?
7. What failure modes or uncertainty does it introduce?
8. How is its authority bounded?
9. Can its effects be observed, reproduced, rolled back and revalidated?
10. Does it make Exagium more capable, more reliable, or both — or merely increase the number of components?

A useful long-term test is whether Exagium can discover a better way to be Exagium, while retaining the ability to establish that the change is genuinely better.

## 17. Summary

Exagium should evolve towards an **evolving, heterogeneous computational discovery environment** that autonomously investigates problems, discovers resources and methods, constructs and composes capabilities, evaluates them through evidence-driven experimentation, and recursively improves the processes through which it discovers and develops new capabilities.

Its architecture should be strong in invariants, explicit interfaces, state semantics, provenance, authority and observability, while remaining open to new internal methods, compositions and computational arrangements.

The mathematical-geometry question is a potentially deep research direction: determine whether common structural constraints and transformations can generate reusable organisation across operational, supervisory and self-improving systems at multiple scales. Treat candidate number sequences and geometries as hypotheses to test, not as architecture by aesthetic selection.

The immediate engineering priority remains to make the existing system coherent and reliable enough to support this direction. The ambition should not be reduced to fit the current implementation, but neither should the current implementation be mistaken for having already achieved the ambition.
