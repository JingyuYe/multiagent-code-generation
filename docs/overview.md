# System Overview & Logic Mapping

This document provides a deep dive into the underlying systems, data flow, and evaluation criteria that power this benchmarking project.

## 1. Data Provenance: The MBPP Benchmark
The system retrieves tasks strictly from the **Mostly Basic Programming Problems (MBPP)** dataset natively hosted on HuggingFace.
- **Implementation Location**: `scripts/run_all_experiments.py` & `scripts/run_mbpp.py`.
- **Flow**: We use the `datasets` library to pull the `sanitized` version of MBPP. For each task, we extract the natural language `prompt` and the hidden `test_list` (a list of Python `assert` strings).
- **Execution Condition**: The agent never explicitly sees the unit tests unless provided as an evaluation control; it must rely on its own reasoning to generate structurally aligned code.

## 2. Model & Inference Engine
We manage LLM inference centrally to enforce strict effort constraints.
- **Model**: We support dual engines. Primarily, we evaluate utilizing lightweight, localized **GPT-5.4 Nano** queries via the OpenAI SDK, which avoids heavy local memory fragmentation. Also included is configuration for Ollama's local `qwen2.5-coder:7b` for 100% offline inference on Apple Silicon.
- **Implementation Location**: `src/core/model.py`. Use the `.env` file (`USE_OPENAI=true`) to toggle.
- **Effort Constraints ($e \in \{1, \dots, n\}$)**: 
  - Effort is not fine-tuned into the model. Instead, it is constrained at the inference stage. 
  - **High Effort ($e=1$)**: Granted a generous generation constraint (`num_predict: 2048`) and slightly higher temperature (`0.6`) for reasoning elasticity.
  - **Low Effort ($e=0$)**: Severely constrained generation (`num_predict: 256`) and deterministic greedy decoding (`temperature: 0.0`), forcing zero-shot "guess" behaviors.

## 3. The Orchestration Framework (LangGraph)
We model interaction through LangGraph `StateGraphs`. Every agent is a Node and shares a universally accessible memory ledger `TaskState` defined in `src/core/state.py`.
- **G0 (Single-Agent Baseline)**: Maps directly from Task -> Output.
- **G1 (Waterfall)**: Linear sequence of Planner -> Coder -> Reviewer -> Tester. No backtracking.
- **G2 (AgentCoder)**: A conditional loop between Coder <-> Tester based purely on test execution success.
- **G2.5 (Reviewer Repair)**: Execution feedback with an injected static analyzer before Coder patch revision.
- **G3 (MapCoder Cycle)**: A complex cycle where test failures trigger a complete rewrite loop originating back at the Planner.

## 4. Evaluation Criteria

### A. Pass@1 Accuracy
Measured by whether the generated string of Python code can pass all hidden assertions on the very first "submit" loop.
- **Implementation**: Evaluated inside the execution sandbox at `src/execution/sandbox.py`. Code is written to a temp file and executed via `subprocess.run()` with a 5-second timeout to safely catch infinite loops and segfaults.

### B. Utility Function ($U_i$)
To study multi-agent code generation under Cooperative Game Theory, we utilize the following equation:
$U_i = V \cdot \mathbf{1}[Y=\text{pass}] - \lambda \cdot T_i$
- **Implementation**: `src/game_theory/utility.py`.
- **Purpose**: We price the token usage ($T_i$, tracked in `TaskState`) against the theoretical value of success ($V$) and token cost modifier ($\lambda$).

### C. Approximate Nash Equilibrium
We test Incentive Stability through an Empirical Best-Responses format.
- **Implementation**: `src/game_theory/stability.py`.
- **Logic**: We run a "Base" high-effort graph. Then we systematically execute "Deviation" graphs where exactly *one* agent is starved to low-effort ($e_i=0$). We compute their utilities. If the deviating agent received a higher utility score by slacking off (minus an $ϵ$ buffer), the topology is deemed **Unstable** since it incentivizes low-effort free-riding.

### D. Shapley Value Estimations
- **Implementation**: `src/game_theory/shapley.py`.
- **Purpose**: Evaluates an agent's marginal contribution ($v_g(S)$) by programmatically silencing them from the graph topology and comparing the Pass@1 drop-off, either exactly (listing all permutations) or through Monte-Carlo sampling.
