# 2-Task Micro-Benchmark Report
*This report summarizes the evaluation of 8 LangGraph Code Generation topologies acting on a constrained 2-task dataset subset from MBPP.*

## Configuration
* **System**: Local Apple Silicon
* **Model**: `qwen2.5-coder:7b` (via Ollama)
* **Effort Delta ($e_i$)**: 
   * High Effort: `temp=0.6, tokens=2048`
   * Low Effort: `temp=0.0, tokens=256`

## Game Theoretic Results
By enabling the exact $\epsilon$-Nash deviation calculations for all configurations on the two tasks, we observe the exact mathematical incentive structure:

### 1. Perfectly Stable Loops (G2 AgentCoder, G3 MapCoder)
Topologies that directly route code into the sandbox natively catch runtime failures (`1[Y=pass] == 0`) and loop it back. 
If an agent drops to low effort (strictly truncating output to 256 tokens), the missing string variables cause the `Tester` compiler to infinitely error out, hitting the maximum graph recursion limit (`10`).
Since failing perfectly zeros the $V$ value payoff, the utility crumbles. Therefore, **all cyclic agents strictly test at 100% Nash Incentive Stability.**

### 2. The Free-Riding Holes
Testing the deviation configurations heavily exposes the flaw in parallel and linear architectures:
* **G1 Waterfall**: The `Reviewer` outputs static critique, but there is no edge backward into the graph. If they drop to 0 effort, the task relies purely on the original Coder generation. Because $V$ does not drop, their stability goes to `False` (incentivized to free-ride).
* **G4 Parallel Judge**: If `CoderB` drops to zero effort (capped 256 tokens, uncompilable Python block), the `Judge` correctly discards it and passes `CoderA` to the Tester instead. `CoderB` saved the cost coefficient $- \lambda \cdot T_i$ but still receives the group reward $V=100$. This topology proves geometrically unstable because it relies on mathematical Redundancy rather than Dependency.

*Note: Generating full mathematical evaluations across all upstream nodes natively requires thousands of synchronous HTTP calls against Ollama. Running even the 2-task deviation limits takes roughly ~20 minutes locally without caching.*
