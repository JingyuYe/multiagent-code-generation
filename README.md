# Multi-Agent Code Generation: Graph Topologies & Incentive Stability

This project evaluates how different interaction graph structures in multi-agent LLM systems affect the tradeoff between code accuracy and compute cost on the MBPP benchmark. It frames agents through Cooperative Game Theory, using Shapley Values to measure marginal contributions and approximate Nash Equilibriums to evaluate incentive stability under constrained token budgets.

## 📊 Latest Results (GPT-5.4 Nano, 50 tasks)

| Topology | Pass@1 | Avg Tokens | Nash Stability |
|:---|:---|:---|:---|
| G0: Baseline (0-shot) | 4.0% | 131 | 100.0% |
| G1: Waterfall (Linear) | 74.0% | 1204 | 84.0% |
| G2: AgentCoder (Loop) | 84.0% | 987 | 78.0% |
| G2.5: Reviewer Repair | 83.7% | 1220 | 81.6% |
| G3: MapCoder (Cycle) | **87.8%** | 1658 | 77.6% |
| G4: Parallel Judge | 72.0% | 4717 | 74.0% |
| G5: Adversarial Debate | 78.0% | 5506 | 26.0% |
| G6: Hierarchical Setup | 74.0% | 7008 | 10.0% |

## 🚀 Setup & Installation

1. **Clone and Setup Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Model:**
   - **GPT-5.4 Nano (recommended):** Set `USE_OPENAI=true` and your `OPENAI_API_KEY` in `.env`.
   - **Ollama (offline):** Start the Ollama daemon and pull the model:
     ```bash
     brew services start ollama
     ollama pull qwen2.5-coder:7b
     ```

3. **Verify Installation:**
   ```bash
   python3 scripts/test_g0.py
   ```

4. **Run Full 50-Task Evaluation:**
   ```bash
   python3 scripts/run_all_experiments.py
   ```

---

## 🏗️ Codebase Structure

```
src/
├── agents/          # Agent nodes: Planner, Coder, Reviewer, Tester, Judge, etc.
├── core/            # Shared state (TaskState), model factory, config
├── execution/       # Subprocess sandbox with 5s timeout
├── game_theory/     # Utility, Nash stability, Shapley value calculations
└── graphs/          # G0-G7 + G2.5 LangGraph topology definitions
scripts/             # Experiment runners & test scripts
utils/               # Notebook generators, image stitching, result parsers
docs/                # Design docs, reports, proposal
data/                # Raw results (.jsonl), CSV summaries, task manifests
```

For a comprehensive overview of data provenance, LLM effort constraints, evaluation metrics, and Game Theory calculations, see [docs/overview.md](./docs/overview.md).

## 📝 Reports
- [Benchmark Report](./docs/report.md) — Full analysis of all topology results
- [Analysis Notebook](./analyze_results.ipynb) — Interactive visualizations (Pass@1, token cost, Nash stability)
- [High-Level Design](./docs/high_level_design.md) — Architecture and development phases
