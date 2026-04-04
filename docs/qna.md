# Q&A Prep Sheet: Graph Topologies and Incentive Stability

This document catalogs an exhaustive list of anticipated questions from professors, graders, and peers regarding the final presentation, along with highly insightful, project-specific answers.

---

## 1. Game Theory & Methodology Mechanics

**Q: Why model LLM interactions using Cooperative Game Theory? Aren't they just static algorithms answering prompts?**
**A:** "While an LLM instance is static, a *Multi-Agent System* is dynamic. In complex structures like parallel chains or hierarchies, agents rely on the output of preceding agents. If an upstream agent generates poor work, downstream agents either fail or must spend massive compute context trying to fix it. Game theory allows us to mathematically quantify 'free-riding'—where an agent slacks off (saving token cost) but still receives the reward because a sibling node carried the weight. It's the only way to prove if an architecture is structurally robust or inherently wasteful."

**Q: How exactly did you constrain 'Effort' ($e \in \{0, 1\}$) for an LLM? You can't just tell an LLM 'try less.'**
**A:** "We controlled effort architecturally at the inference layer. High effort ($e=1$) was granted a generous token generation limit (`max_tokens: 2048`) and allowed step-by-step reasoning prompts. Low effort ($e=0$) was bottlenecked by a strict `max_tokens` limit (e.g., 256) and greedy deterministic decoding (`temperature: 0.0`), forcing it to output zero-shot unreflective guesses. This cleanly simulated the delta between an agent taking time to verify its work versus throwing a lazy answer over the wall."

**Q: How did you compute Shapley Values? Computing every permutation for a multi-node graph is computationally explosive.**
**A:** "Exactly. Evaluating $2^N$ subset combinations across 50 tasks with real LLM inference would be impossible due to cost and API limits. Instead, we used a Monte-Carlo sampling of agent arrival permutations ($\pi$) to approximate the marginal contributions ($\phi_i(g)$). Furthermore, when an agent was 'silenced' from a coalition (e.g., bypassing the Reviewer), the LangGraph orchestrator dynamically rerouted the edges to skip that node entirely, allowing us to measure the raw Pass@1 drop-off of its absence."

**Q: In your Game Theory formulation, why was $\lambda$ hard-coded, what is its exact value, and how do you justify it?**
**A:** "In our `src/game_theory/utility.py` file, we strictly hardcoded $V = 100$ (the reward for passing) and $\lambda = 0.01$ (the penalty scalar per token). We justified $\lambda = 0.01$ because it mathematically zeroes out the task reward exactly at 10,000 tokens ($100 - (0.01 \times 10,000) = 0$). This establishes a hyper-realistic bound: if a single agent burns 10,000 tokens trying to pass a basic Python unit test, the cost of inference eclipses the value of the software itself. It properly punishes topologies like G6 Hierarchical, where a 'Manager' generates close to 7,000 tokens of context bloat, thus dragging their Utility score down by 70 points and mathematically proving their inefficiency."

**Q: The Utility function subtracts Token Cost ($\lambda \cdot T_i$). What happens if API costs drop to near-zero in the future? Does your Nash stability metric break down?**
**A:** "It doesn't break down; it restricts the equilibrium. If $\lambda \to 0$, token cost ceases to be a penalty. In that scenario, all agents are mathematically incentivized to exert maximum effort constantly, and chaotic topologies like Adversarial Debate (G5) become 'free' to run. However, compute is never truly infinite. In real-world environments, $\lambda$ can represent *latency* (Time-to-First-Token) rather than literal dollars. As long as users care about latency, complex redundant graphs will always penalize the utility."

**Q: You use an $\epsilon$-Nash Equilibrium buffer in your inequality. Why an approximate Nash rather than a strict Nash Equilibrium?**
**A:** "LLMs are inherently stochastic. Even with temperature set to 0.0, the exact token consumption ($T_i$) will fluctuate slightly across runs due to API formatting or trace variations. By using an epsilon ($\epsilon$) buffer, we ensure that an agent deviation is only flagged as 'incentivizing free-riding' if the Utility gained by slacking off is statistically larger than natural LLM token flux. It prevents false-positives in our instability metrics."

**Q: Standard Shapley values assume the characteristic function $v(S)$ is 'superadditive' (adding members increases the value). Does that hold true in LLM graphs?**
**A:** "That assumption actually breaks down in LLM systems, which is exactly why calculating the marginal contribution is so revealing! Unlike classical economics where a new worker rarely decreases output, adding a hallucinating 'Judge' (G4) or a hyper-critical 'Red Team' (G5) can actively overwrite correct code and cause task failure. Agents that routinely decrease the Pass@1 rate when joining a coalition will mathematically yield negative or zero Shapley values, explicitly proving that specific structural node is detrimental."

**Q: Why model this bridging both Cooperative (Shapley) and Non-Cooperative (Nash) game theory?**
**A:** "It gives us a holistic 360-degree view. We use Cooperative theory (Shapley) to figure out who actually deserves the credit when the group globally wins the $V=100$ reward. Conversely, we use Non-Cooperative theory (Nash) to evaluate the system strictly from a selfish perspective: if an agent only cares about minimizing its own token penalty $-\lambda T_i$, will it betray the pipeline? Bridging both frameworks reveals if a topology perfectly aligns selfish efficiency with cooperative success."

**Q: Could an 'unstable' topology actually be Pareto Optimal for the agents?**
**A:** "Absolutely, and that's the danger we are highlighting. In the Hierarchical model (G6), if a worker agent slacks off, they drastically increase their own Utility by saving tokens. As long as the *other* parallel workers still stumble into a passing answer, the group reward $V$ is preserved. Because the slacking agent improves their status without hurting the others' payouts, that lazy state is technically Pareto Preferred for the free-rider. That proves the architecture structurally misaligns the agents away from high-effort verification."

**Q: Have you considered modeling this as a repeated game rather than a one-shot game? The agents run across 50 tasks.**
**A:** "Great question. In a repeated game, agents could theoretically learn to 'punish' free-riders in future rounds — like a tit-for-tat strategy. However, our LLM agents have no persistent memory between tasks. Each MBPP problem starts with a fresh `TaskState`. The Coder on Task 12 has zero knowledge of what happened on Task 11. This makes each invocation a genuinely independent, one-shot game, which is why modeling it as a one-shot Nash equilibrium test per task is the correct formulation."

**Q: Is the solution concept you're testing closer to a Dominant Strategy Equilibrium or a Nash Equilibrium?**
**A:** "It's strictly a Nash Equilibrium test, not dominant strategy. A dominant strategy would mean high effort is optimal *regardless* of what other agents do. In our system, that's not always true — in G1 Waterfall, if the Coder writes perfect code, the Reviewer's effort is genuinely irrelevant (their review doesn't feed back anywhere). So the Reviewer doesn't have a dominant strategy to work hard. Nash is the right lens because it captures these topology-dependent conditional incentives."

**Q: Your reward $V$ is shared equally. Is that realistic? What about asymmetric payoffs?**
**A:** "We deliberately chose a shared, symmetric reward because in an LLM pipeline, there is no natural mechanism to price-discriminate agent contributions *before* task completion. The code either passes or fails as a unit. However, this is exactly what makes Shapley values so critical — they retroactively decompose the shared $V=100$ reward into fair, contribution-weighted allocations *after the fact*. So the reward is shared at payout time, but Shapley tells you who actually earned it."

**Q: Could you use a mechanism design approach — like a VCG (Vickrey-Clarke-Groves) auction — to incentivize truthful high effort?**
**A:** "In theory, yes. A VCG mechanism would charge each agent a tax proportional to the externality they impose on others. If a lazy Coder causes the Tester to burn 2,000 extra tokens debugging, VCG would subtract that cost from the Coder's utility. The challenge is that LLM agents don't have monetary budgets or true preferences — we'd need to translate VCG payments into inference constraints, which is an exciting future research direction."

**Q: How does your work relate to the classic Principal-Agent problem in economics?**
**A:** "It maps directly. The *Principal* is the system designer (us) who wants maximum Pass@1 accuracy. The *Agents* are the LLM nodes who bear the token cost of effort. The classic moral hazard problem arises: the Principal cannot directly observe an agent's 'effort level' — only its output. In our framework, the Nash deviation sweep acts as an audit mechanism. By toggling effort and measuring utility changes, we reverse-engineer whether the topology's structure itself creates moral hazard."

**Q: What is the 'Price of Anarchy' in your system? How much performance is lost when agents act selfishly vs. cooperatively?**
**A:** "We can calculate it directly from our data. The Price of Anarchy is the ratio of optimal social welfare to the worst-case Nash equilibrium welfare. In G3 MapCoder, the gap is small — selfish high effort and cooperative high effort converge because every agent is structurally dependent on the others (the cycle forces collaboration). But in G6 Hierarchical, the Price of Anarchy is enormous: the socially optimal outcome requires all workers to contribute, but the Nash equilibrium allows workers to free-ride at 10% stability, causing a massive welfare loss."

**Q: Could you apply mixed strategies — where agents randomize between high and low effort — instead of pure strategies?**
**A:** "Mixed strategies would model an agent choosing high effort with probability $p$ and low effort with probability $1-p$. While theoretically interesting, it's impractical for LLM systems because you can't meaningfully randomize an LLM's effort — you either give it 2048 tokens or you don't. There's no continuous slider. Our binary effort constraint ($e \in \{0, 1\}$) naturally restricts the analysis to pure strategies, which is actually more realistic for deployed API systems where you set a fixed `max_tokens` parameter."

**Q: Does the 'Core' of this cooperative game exist? Is the grand coalition (all agents working together) stable?**
**A:** "The Core is the set of payoff distributions where no subset of agents would prefer to break away and form their own coalition. In our cyclic topologies (G2, G3), the Core exists because every agent is structurally essential — remove the Coder and you get 0% accuracy, remove the Tester and you lose the execution feedback loop. No subset can do better alone. But in G4 Parallel Judge, the Core is empty: any single Coder could break away, skip the Judge entirely, and achieve comparable accuracy at lower cost. An empty Core mathematically proves the topology is coalitionally unstable."

---

## 2. Topology Architectures & Behaviors

**Q: Why did MapCoder (G3) so heavily dominate the other complex architectures (87.8%)?**
**A:** "MapCoder forces a *global* rewrite cycle rather than a *local* patch. Because test failures and Reviewer logic trigger a route all the way back to the **Planner**, the system reconstructs the logical algorithm from scratch using the fail trace. Topologies like AgentCoder (G2) just patch syntax locally. More importantly, MapCoder relies on *literal execution feedback* (the Sandbox) combined with static analysis. Execution feedback is mathematically perfect and strictly cheaper than LLM analysis alone."

**Q: Your data shows that Hierarchical (G6) and Adversarial Debate (G5) collapsed in Nash Stability (10-26%). Why?**
**A:** "Because of the *Parallelism Trap*. In G6, a Manager broke down simple MBPP logic puzzles into parallel worker subtasks. The context overhead required for the Aggregator to stitch fragmented code back together was enormous (averaging 7,000+ tokens). Because the task was effectively duplicated, an individual worker could drop to 'Low Effort' and the Aggregator was often still able to cobble a passing script from the others. The agent conserved its token cost penalty ($-\lambda \cdot T_i$) but still received the group payoff ($V$), thus proving the architecture incentives free-riding."

**Q: In your Parallel Judge (G4) topology, why didn't the Judge just pick the best code?**
**A:** "The Judge suffers from the limitations of static LLM analysis. It evaluates three Python strings blindly. Without a Python interpreter to execute them, the Judge frequently hallucinated, couldn't distinguish edge cases, and often picked an elegant-looking script that contained a fatal runtime syntax error. It proves that redundant generation without execution validation is a token-burning trap."

**Q: What happens if the cyclic topologies (AgentCoder, MapCoder) get stuck in an infinite debugging loop?**
**A:** "We enforced a strict graph recursion limit (e.g., $K=50$ loops) natively within LangGraph. If the Coder repeatedly failed the unit tests and exhausted the cycle limit, the LangGraph `pregel` engine threw a RecursionError, which we caught and graded as a `[Y=fail]`. This ensures the system doesn't burn infinite tokens on an unsolvable task."

**Q: I noticed in your methodology you made a distinction between G2 AgentCoder and G2.5 Reviewer Repair. What was the exact role of the Reviewer?**
**A:** "We actually discovered a critical architectural bug during evaluation where the `Reviewer` node was generating brilliant static analysis, but the downstream `Coder` and `Planner` prompts weren't explicitly ingesting the `review_comments` from the `TaskState`. Once we explicitly piped that feedback into G2.5 and G3, their Pass@1 skyrocketed, proving that routing data correctly through the graph is just as important as the nodes themselves."

---

## 3. Evaluation & Experimental Design

**Q: Why did you use the MBPP dataset instead of a harder benchmark like SWE-Bench or HumanEval?**
**A:** "MBPP was necessary for computational feasibility. To properly establish empirical Best-Responses for Nash Equilibriums, we had to run deviation sweeps (starving exactly one agent at a time across multiple nodes) on the exact same task. MBPP allowed us to hit statistical significance on evaluating *topology interaction* without conflating failures with insurmountable task difficulty. If the task is too hard (SWE-Bench), all topologies fail and the utility metrics flatline."

**Q: How did you handle infinite loops generated by the LLM during evaluation?**
**A:** "We isolated the compiled Python strings into temporary files and executed them via a `subprocess` wrapper equipped with a hard 5-second timeout. If a looping agent (like AgentCoder) generated a `while True:` loop, the sandbox gracefully caught the `TimeoutExpired` exception, terminated the child process, and fed the timeout error string backward into the graph as explicit negative feedback."

**Q: In MBPP, the unit tests are provided. In the real world, test cases must be written by the developer. How would this affect your findings?**
**A:** "This is a fantastic extension of our research. If we added a `TestDesigner` node (which AgentCoder traditionally uses), the accuracy would likely drop because LLMs struggle to write perfect, bug-free unit tests. If the tests themselves are flawed, the Sandbox execution feedback becomes polluted, which would severely degrade the efficiency of our winning cyclic topologies (G3)."

---

## 4. Broader Implications

**Q: If you had to build a production multi-agent coding system for a tech company tomorrow based on this research, what would it look like?**
**A:** "We would build a variant of **G3 (MapCoder)** integrated with **G2.5 (Reviewer Repair)**. We would completely ban horizontal peer-debate (G5) and redundant 'Best-of-N' generation (G4) to save 75% on API costs. The pipeline would be: Planner -> Coder -> Sandbox. If it fails, send the literal Stacktrace to a Reviewer for static analysis, and feed the Reviewer's report back to the Planner for a fresh architectural pass. Strict, execution-backed, vertical cycles."

**Q: You evaluated this using GPT-5.4 Nano and Qwen2.5-Coder. Do you expect these incentive stability results to hold for much larger models like GPT-4o or Claude 3.5 Sonnet?**
**A:** "Yes, the mathematical principles of Game Theory apply regardless of the foundation model's parameter size. While a smarter model might increase the absolute Pass@1 rate across all topologies, the *relative* token cost and the marginal contribution (Shapley value) of a redundant agent (like a hallucinating Judge) will still penalize the Utility function. A smarter model makes the baseline higher, but it doesn't fix a broken interaction graph."

**Q: Why exactly did the Manager/Worker (G6) model fail on MBPP but succeed in frameworks like MetaGPT?**
**A:** "Frameworks like MetaGPT are designed for massive software engineering tasks (like building an entire Flappy Bird game with CSS, HTML, and JS). MBPP consists of single-function algorithmic puzzles (e.g., 'Find the volume of a triangular prism'). Forcing a Manager to decompose a 5-line function into 3 separate worker threads creates artificial fragmentation. It proves that topology effectiveness is strictly bounded by the *complexity of the dataset*."
