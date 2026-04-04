# Presentation Speaker Notes
**Target Duration:** 10 minutes (~45 seconds per slide)
**Focus:** Connecting interaction topologies, empirical performance, and game-theoretic mathematics.

---

### Slide 1: Title Page
**Speaker Notes:**
"Hello everyone. Today, my team—Charlotte, Nana, and I—are presenting our research on 'Graph Topologies and Incentive Stability in Multi-Agent Code Generation.' Over the next 10 minutes, we’ll explore how structurally organizing LLM agents changes not just their coding accuracy, but their mathematical incentives to expend effort."

### Slide 2: Motivation
**Speaker Notes:**
"Multi-agent AI is rapidly taking over software engineering, but it comes with a massive, compounding cost in API tokens and latency. While decomposing tasks into roles—like a discrete Coder and Tester—reduces syntax errors, recent structural analyses like Lyu et al. show that agent failures often trace directly to inter-agent miscommunication. Therefore, the single most critical, yet understudied, hidden variable in LLM orchestration is the *Interaction Topology* itself—the literal shape of how agents speak to one another."

### Slide 3: Background
**Speaker Notes:**
"Looking at the evolution of code generation: We moved from single-shot guessing with early GPT models to formal pipeline orchestration like AutoGEN, and finally to modern closed-loop paradigms like AgentCoder or MapCoder, which loop test failures back to the LLM. 
However, the *Gap* is that current research evaluates these frameworks as holistic black boxes—changing the prompts, tools, and models simultaneously. We set out to isolate the *Topology itself* as the independent variable to see how structural arrangement drives token economics."

### Slide 4: Overview: Our Approach
**Speaker Notes:**
"To solve this, we combined LLM orchestration with Cooperative Game Theory. We built a framework to rigorously test 8 distinct graph structures using the same underlying LLM. By systematically choking the 'effort' given to individual agents and tracking the token cost, we can measure how susceptible a given architecture is to 'free-riding' and structural bloat."

### Slide 5: Game Theory Formulation
**Speaker Notes:**
"We model multi-agent coding as a cooperative game. In this figure/concept, all agents share a global objective: passing the MBPP unit tests. 
The core math is our Utility Function: $U_i = V \cdot \mathbf{1}[Y=\text{pass}] - \lambda \cdot T_i$. 
Here, $V$ is the shared reward for a passing script. But crucially, we subtract $\lambda \cdot T_i$, which is the exact Token Cost expended by agent $i$. This forces agents into a mathematical tradeoff: do I spend expensive tokens verifying my code, or do I slack off and hope my peers fix it?"

### Slide 6: Operational Framework: Agent Roles and Effort Constraints
**Speaker Notes:**
"Here we define our agent nodes and what they can play. We have standard roles: Planner, Coder, Reviewer, and Tester. 


But since we cannot 'fine-tune' effort out of an API, we manipulate Effort ($e$) using generation constraints. High effort ($e=1$) gives the agent 2048 tokens and reasoning prompts. Low effort ($e=0$) starves the agent—capping it at 256 tokens and greedy, 0-temperature decoding. This perfectly simulates an agent throwing lazy, unverified code over the wall."

### Slide 7: Graph Topologies Overview
**Speaker Notes:**
"These are the exact topologies we evaluated, built in LangGraph. I'll briefly break down how information flows through each one:
- **G0: Baseline (0-shot):** A single Universal Agent mapping input directly to output with no reflection. *(Inspiration: standard ChatGPT zero-shot inference).*
- **G1: Waterfall:** A strictly linear sequence of Planner, Coder, Reviewer, and Tester. There is no backtracking allowed. *(Inspiration: Traditional waterfall SDLC).*
- **G2: AgentCoder:** A tight, conditional loop between Coder and Tester. It iterates dynamically based purely on test execution success. *(Inspiration: AgentCoder, Huang et al., 2023).*
- **G2.5: Reviewer Repair:** Routes test failures through a Reviewer for static analysis *before* returning to the Coder for patching. *(Inspiration: Bridging Reflexion, Shinn et al., 2023, with AgentCoder).*
- **G3: MapCoder:** A massive cyclic flow where test failures trigger a complete re-plan originating all the way back at the Planner, rigorously rebuilding logical rot. *(Inspiration: MapCoder, Islam et al., 2024).*
- **G4: Parallel Judge:** A 'Best-of-N' structure where three parallel Coders produce candidate solutions, and a Judge votes on the best one to forward to the Tester. *(Inspiration: AlphaCode scaling laws and Best-of-N sampling).*
- **G5: Adversarial Debate:** The Coder's script is attacked by a 'Red Team' searching for vulnerabilities, and a 'Blue Team' must defend or patch it before testing. *(Inspiration: Multi-agent debate frameworks / Du et al., 2023).*
- **G6: Hierarchical Setup:** A Manager agent breaks down tasks and hands them off to parallel Workers, followed by an Aggregator. *(Inspiration: MetaGPT and ChatDev organizational trees).*"

### Slide 8: Implementation
**Speaker Notes:**

"For implementation, we used the LangGraph framework to strictly orchestrate the edges, allowing us to swap out GPT-5.4 Nano and local Qwen instances cleanly. 
We attached telemetry callbacks to precisely count every prompt and completion token for that $T_i$ utility calculation. For evaluation, we ran exactly 50 logic tasks from the MBPP dataset, executing the generated Python in a secure, subprocess sandbox with a 5-second timeout to catch infinite loops natively."

### Slide 9: Evaluation: Shapley Values
**Speaker Notes:**
"Our first evaluation metric is Shapley Values ($\phi_i$). We use this to answer a classical cooperative game theory question: 'How much credit does a specific agent—like the Reviewer—truly deserve for a successful execution pass?' 

The formal equation for Shapley values calculates an agent's marginal contribution: 
$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n - |S| - 1)!}{n!} (v(S \cup \{i\}) - v(S))$

In plain terms:
1. $v(S \cup \{i\}) - v(S)$ measures the literal Pass@1 accuracy gained when agent $i$ joins a coalition of other agents $S$. If adding a 'Judge' actively breaks the code, their contribution is negative.
2. The factorial fraction $\frac{|S|! (n - |S| - 1)!}{n!}$ accounts for every possible permutation of agents arriving to do the task. 

However, testing all $2^N$ combinatorial pipelines per MBPP task with massive 7-billion parameter language models is computationally explosive. Therefore, we utilize **Monte-Carlo sampling of agent arrival permutations**. Instead of running every combination, we randomly 'silence' specific agents from the LangGraph orchestrator—bypassing them entirely—and sample the resulting Pass@1 drop-off to statistically approximate that node's true economic worth."

### Slide 10: Evaluation: Nash Equilibrium Test
**Speaker Notes:**
"Our second, and most critical, metric is Incentive Stability. To test if a high-effort profile is a Nash Equilibrium, we run an empirical deviation sweep: $U_i(e_i=1, e_{-i}^*) \geq U_i(e_i=0, e_{-i}^*) - \epsilon$. 
In English: We starve exactly one agent to low effort ($e=0$) while the group runs high effort. If the starved agent achieves a *higher* utility score because they saved token costs ($-\lambda \cdot T_i$) but the team still successfully passed the task ($V$), the agent was incentivized to free-ride. The topology is therefore unstable."

### Slide 11: Results
**Speaker Notes:**
"Our empirical results are striking. As seen in the visualizations, **Cyclic topologies (G2, G3) vastly out-scaled everything else.** G3 MapCoder hit an **87.8% Pass@1 accuracy** while maintaining an incredible 77% Nash Stability. 
Conversely, complex horizontal structures like **G6 Hierarchical** and **G4 Parallel** were traps. They burned between 4,000 and 7,000 tokens per loop, dragging their Utility down. Because redundant workers could easily 'slack off' while the Manager still cobbled a passing script together, G6’s Nash Stability collapsed to a catastrophic 10%."

### Slide 12: Conclusion
**Speaker Notes:**
"Our game-theoretic findings lead to two major conclusions:
First, **The Supremacy of Execution over LLM Judgment.** Frameworks that rely on LLM 'Debates' or 'Judges' burn thousands of tokens hallucinating over semantics. Frameworks that simply compile the code in a sandbox and feed the stack-trace back to the Coder achieve near-perfect pass rates for a fraction of the cost.
Second, **The Parallelism Trap.** Over-fragmenting a task into deep managerial hierarchies heavily dilutes an individual agent's marginal contribution, devastating their Nash incentive stability and encouraging free-riding."

### Slide 13: Discussion & Limitation
**Speaker Notes:**
"As a limitation, our evaluation was conducted on MBPP—which consists of algorithmically simple, single-function Python puzzles. This simplicity is exactly why the G6 Manager/Worker structure struggled with extreme context bloat. 
For future work, expanding this Nash Equilibrium evaluation to broad, multi-file software engineering tasks like SWE-bench would determine if the 'Parallelism Trap' disappears when the sheer complexity of the task finally justifies a multi-worker hierarchy."

### Slide 14: Q&A
**Speaker Notes:**
"Thank you for your time. The code, sandbox implementation, and raw results dataset are available in our repository. We’ll now open the floor to any questions."
