# High-Level Implementation Design: Graph Topologies and Incentive Stability in Multi-Agent Code Generation

## 1. System Overview
The system is designed to orchestration multi-agent coding workflows, execute generated code against the MBPP benchmark, and analyze the results using Cooperative Game Theory metrics (Shapley Values and Utility/Nash Equilibriums). 

The implementation relies on **LangGraph** for flexible orchestration of different graph topologies and **LangChain**/**OpenAI SDK** to interface with **GPT-5.4 Nano** (primary) and Ollama's local `qwen2.5-coder:7b` (offline fallback).

## 2. Core Components

### 2.1 Agent Definitions
We will define four distinct agents, each with specific system prompts and responsibilities:
- **Planner (PL)**: Breaks down the MBPP task into a step-by-step logic plan.
- **Coder (C)**: Translates the logic plan directly into Python code.
- **Reviewer (R)**: Statically analyzes the generated Python code for logical errors, edge cases, and style prior to execution.
- **Tester (T)**: Executes the generated unit tests, reviews execution traces, and isolates runtime errors.

### 2.2 Effort Level Modulation ($e \in \{0, 1\}$)
Because the foundation model (GPT-4o) remains fixed without fine-tuning, effort will be artificially constrained:
- **High Effort ($e=1$)**: 
  - Uses highly detailed prompts (e.g., "Think step-by-step, consider 3 edge cases, rigorously double check logic").
  - Provided a larger `max_tokens` generation budget.
  - Allowed multiple internal sampling/reflection rounds.
- **Low Effort ($e=0$)**: 
  - Uses minimal prompts (e.g., "Provide a quick, concise answer. Do not explain.").
  - Tight `max_tokens` constraint.
  - 0-shot generation logic.

### 2.3 Graph Topologies (LangGraph StateGraphs)
We implement nine discrete `StateGraph` configurations:
- **G0: Single-Agent Baseline**: Uses a single Universal Agent mapping `input -> output` with optional self-check.
- **G1: Waterfall Pipeline**: Strictly linear sequential flow: `Task -> PL -> C -> R -> T -> Output`.
- **G2: AgentCoder-Style Loop**: Iterative flow branching at Tester: `Task -> C <---> T`. Loops until tests pass or `$K$` cycles are exhausted.
- **G2.5: Reviewer Repair**: Extends G2 by routing test failures through a Reviewer for static analysis before cycling back to Coder: `Tester -> Reviewer -> Coder`.
- **G3: MapCoder-Style Cycle**: Complex cyclic flow where test failures trigger a complete re-plan originating at the Planner, incorporating reviewer comments.
- **G4: Parallel Judge (Best-of-N)**: Three parallel Coders produce candidates, a Judge selects the best, forwarded to Tester.
- **G5: Adversarial Debate**: A Red Team attacks the code and a Blue Team defends/patches it before testing.
- **G6: Hierarchical Setup**: A Manager decomposes the task into sub-tasks assigned to parallel Workers, aggregated by an Aggregator.
- **G7: Mental Simulator**: A Simulator node mentally traces execution before the Tester runs real tests.

## 3. Evaluation & Execution Engine

### 3.1 MBPP Benchmark Loader
Load the MBPP dataset using the `datasets` library. Filter to a representative, computationally feasible subset for Shapley Value Monte-Carlo approximations to save API costs.

### 3.2 Secure Code Execution Environment
- Execute generated Python code via `subprocess.run()` with a hard 5-second timeout, writing to a temp file to isolate infinite loops and segfaults.
- Capture `stdout`, `stderr`, and runtime exceptions to feed backward into iterative topologies (G2, G2.5, G3).
- Return binary outcome $Y \in \{\text{pass}, \text{fail}\}$.

### 3.3 Token & Cost Tracing
Wrap all LLM calls with a callback handler to precisely track prompt tokens, completion tokens, and interaction rounds per agent. This is strictly required for the constraint equation $T_i$ and the overall Utility calculation.

## 4. Game Theoretic Analysis Engine

### 4.1 Utility Calculation
A module to compute Utility vectors for a given execution trace:
$$U_i = V \cdot \mathbf{1}[Y=\text{pass}] - \lambda \cdot T_i$$
Where $V$ (value of task completion) and $\lambda$ (cost scalar) are configurable hyperparameters.

### 4.2 Incentive Stability (Nash Equilibrium Test)
To test if high-effort profile $e^* = (1,1,1,1)$ is an approximate Nash equilibrium:
1. Run evaluation with all agents at high effort to find base utility.
2. Iteratively set exactly one agent $i$ to low effort ($e_i=0$) while keeping others at high effort ($e_{-i}^*$).
3. Compare utility change: Check if $U_i(e_i=1, e_{-i}^*) \geq U_i(e_i=0, e_{-i}^*) - \epsilon$.

### 4.3 Shapley Value Attribution
- **Subset Power Set**: Implement a mechanism to execute workflows with "missing" agents (e.g. bypassing the Planner, or bypassing the Reviewer) to measure $v_g(S)$.
- **Monte Carlo Permutation**: Because evaluating all subset permutations $2^N$ might be expensive across MBPP, use Monte Carlo sampling of agent arrival permutations $\pi$ to calculate marginal contributions $\phi_i(g)$.

## 5. Development Phases
1. **Phase 1: Foundations** ✅: Implement Agent Prompts, Effort Modulation, Execution Sandbox, and MBPP loader.
2. **Phase 2: Graph Topologies** ✅: Build G0 through G7 (including G2.5) using LangGraph and verify basic pass@1 performance.
3. **Phase 3: Tracing & Utilities** ✅: Add callback systems to log token consumption and measure Utility functions.
4. **Phase 4: Game Theory Evaluation** ✅: Implement empirical best-response iteration and Shapley Monte-Carlo permutation sweeps.
5. **Phase 5: Bug Fixes & Validation** ✅: Fixed critical bug where `review_comments` were generated but silently dropped before reaching Coder/Planner/Tester nodes. Reran affected topologies (G1, G2.5, G3) on the full 50-task evaluation set.
