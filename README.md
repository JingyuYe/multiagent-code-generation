# Multi-Agent Code Generation: Graph Topologies & Incentive Stability

This project evaluates how different interaction graph structures in multi-agent LLM systems affect the tradeoff between code accuracy and compute cost on the MBPP benchmark. It frames agents through Cooperative Game Theory, using Shapley Values to measure marginal contributions and approximate Nash Equilibriums to evaluate incentive stability under constrained token budgets.

## 🚀 Setup & Installation

1. **Clone and Setup Virtual Environment:**
   Run the following to initialize a clean python environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   This project relies on OpenAI's `gpt-4o`. Ensure you have an active `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

3. **Verify Installation:**
   Run the baseline test snippet to verify the LangGraph agents can execute against the sandbox.
   ```bash
   python3 scripts/test_g0.py
   ```

4. **Run MBPP Evaluation:**
   Execute a sample run against the HuggingFace MBPP dataset using the G3 MapCoder Cycle loop:
   ```bash
   python3 scripts/run_mbpp.py
   ```

---

## 🏗️ Codebase Structure & Logic Mapping

The project separates the LLM definition, orchestration (Graph topologies), code execution, and Game Theory calculations into separate modules.

### **1. System Configurations (`src/core/`)**
- **`model.py`**: Contains the `get_llm(effort_level)` definition. **This handles the Effort Level Modulation ($e \in \{1, 0\}$)**. It modifies the prompt logic, `temperature`, and `max_tokens` for high vs. low effort configurations.
- **`state.py`**: Defines the shared `TaskState` dictionary passed between nodes in the graph. **Modifying this file allows you to track new metrics** (e.g. tracking specific API calls or specific subsets of agent errors).

### **2. The Agents (`src/agents/`)**
Contains the `Planner`, `Coder`, `Reviewer`, and `Tester` node definitions. 
- **What to modify:** You will need to tweak the `ChatPromptTemplate` strings here if you want to experiment with different agent prompting strategies (e.g., giving the Planner few-shot examples or isolating the test assertions from the Coder to simulate blind environments).

### **3. Orchestration Graphs (`src/graphs/`)**
This is where the LangGraph topologies are hard-coded.
- `g0_baseline.py`: Single Universal node mapping `Input -> Output`.
- `g1_waterfall.py`: Sequential pipeline `PL -> C -> R -> T`.
- `g2_agentcoder.py`: Branching loop `(PL -> C <-> T)` looping for $K$ iterations if tests fail.
- `g3_mapcoder.py`: Complete cyclic loop `(PL -> C -> R -> T -> PL)`.
- **What to modify:** If you want to introduce a new graph structure (e.g., a "Debate" structure between two Reviewers), you must create a new file here using `StateGraph`.

### **4. Execution Engine (`src/execution/`)**
- **`sandbox.py`**: The environment where the Coder's python string is blindly executed containing the generated tests. **If you need stricter execution security or timeouts (e.g., infinite loops in generated code)**, you should swap the naive `exec()` wrapper here with a `subprocess` or Docker container execution.

### **5. Game Theory Utilities (`src/game_theory/`)**
- **`utility.py`**: Computes agent utility $U_i$ based on $V$ (success value) and $\lambda$ (cost scalar).
- **`shapley.py`**: Computes the exact and Monte-Carlo Permutation metrics for marginal contribution estimation. 
- **What to modify (CRITICAL FOR PROJECT)**: The `stability.py` script is currently a scaffold. You need to implement the Empirical Best-Response test here to find the Approximate Nash Equilibrium. This will involve iteratively "forcing" one agent to $e=0$ (via `model.py`) while leaving all others at $e=1$, computing the new graphs, and tracking the differential $\Delta U_i$.

## 🎯 Next Steps for Project Completion
1. **Implement Nash Equilibrium Test**: Create the loop in `src/game_theory/stability.py` that iterates through the graph with a modified effort profile vector to check the condition: 
   $$U_i(e_i=1, e_{-i}^*) \geq U_i(e_i=0, e_{-i}^*) - \epsilon$$
2. **Batch MBPP Subsets**: Update `scripts/run_mbpp.py` to process larger blocks of the MBPP dataset reliably rather than `test[:2]`.
3. **Data Logging**: Write a CSV or JSON exporter in the script loop to serialize token budgets, $Y$ execution outcomes, and Shapley values over a broad sweep of tasks.
