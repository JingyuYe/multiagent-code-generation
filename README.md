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

2. **Configure Local Model Infrastructure:**
   This project relies on the ultra-powerful 7B coder from Alibaba running locally. Start the Ollama daemon and pull the model:
   ```bash
   brew services start ollama
   ollama pull qwen2.5-coder:7b
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

## 🏗️ Codebase Structure

For a comprehensive overview of how data provenance, LLM effort constraints, evaluation metrics, and Game Theory calculations are mapped, please read [overview.md](./overview.md).

## 🎯 Next Steps for Project Completion
1. **Extend the Sandbox**: The execution environment (`src/execution/sandbox.py`) currently uses a mock `exec()` runtime. Before running untrusted MBPP code across broad datasets, implement a secure containerized shell context. 
2. **Execute Broad Sweeps**: `scripts/run_all_experiments.py` is configured with a tiny limit (`test[:2]`) to test quickly. Increase this limit locally to evaluate over the true 974 task length.
