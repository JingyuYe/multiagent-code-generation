import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """\
# ECE 752 Analysis: Graph Topologies and Incentive Stability in Multi-Agent Code Generation

This notebook analyzes the results of evaluating different multi-agent graph topologies for code generation, based on the proposed methodology. We structure the analysis into three parts based on the ECE 752 research project proposal:
1. **Pass@1 Accuracy vs Topology**
2. **Compute Cost (Tokens) vs Accuracy** 
3. **Incentive Stability (Nash Equilibrium)**
"""

code_imports = """\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
df = pd.read_csv('data/results.csv')
print(df.head(10))
"""

text2 = """\
## 1. Accuracy (Pass@1) by Graph Topology

We explore how different collaborative configurations impact raw performance on the MBPP dataset.
"""

code_acc = """\
# Convert Pass_Rate from string to float
df['Pass_Rate_Float'] = df['Pass_Rate'].str.rstrip('%').astype('float') / 100.0

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Pass_Rate_Float', y='Topology', palette='viridis')
plt.title('Pass@1 Accuracy by Graph Topology', fontsize=16)
plt.xlabel('Pass@1 Accuracy', fontsize=14)
plt.ylabel('Topology', fontsize=14)
plt.tight_layout()
plt.show()
"""

text3 = """\
## 2. Accuracy vs Compute Cost (Tokens)

Next, we look at the tradeoff between how many tokens a topology consumes on average and its resulting accuracy.
"""

code_cost = """\
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Avg_Tokens', y='Pass_Rate_Float', hue='Topology', s=200, palette='deep')

# Annotate points
for idx, row in df.iterrows():
    plt.text(row['Avg_Tokens'] + 50, row['Pass_Rate_Float'], row['Topology'].split(':')[0], 
             horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.title('Pass@1 Accuracy vs Average Token Cost', fontsize=16)
plt.xlabel('Average Tokens per Task', fontsize=14)
plt.ylabel('Pass@1 Accuracy', fontsize=14)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""

text4 = """\
## 3. Incentive Stability (Approximate Nash Equilibrium)

We investigate whether complex graph structures discourage high-effort verification from agents due to varying incentives. We evaluate the percentage of tasks where the fully cooperative, high-effort profile is a Nash Equilibrium.
"""

code_nash = """\
# Convert Nash_Stability from string to float
df['Nash_Stability_Float'] = df['Nash_Stability'].str.rstrip('%').astype('float') / 100.0

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Nash_Stability_Float', y='Topology', palette='magma')
plt.title('Incentive Stability (% of tasks maintaining Nash Eq)', fontsize=16)
plt.xlabel('Nash Stability Rate', fontsize=14)
plt.ylabel('Topology', fontsize=14)
plt.tight_layout()
plt.show()
"""

text_graphs = """\
## Visual Flow of Graph Topologies

Below are the interaction sequence and agent roles for each evaluated topology:

**G0: Baseline (0-shot)**
```mermaid
graph LR;
    P[Problem] --> UA(Universal Agent);
    UA --> O[Output];
```

**G1: Waterfall (Linear)**
```mermaid
graph LR;
    PL(Planner) --> C(Coder);
    C --> R(Reviewer);
    R --> T(Tester);
    T --> O[Output];
```

**G2: AgentCoder (Loop)**
```mermaid
graph LR;
    PL(Planner) --> C(Coder);
    C --> T(Tester);
    T -- Test Fails --> C;
    T -- Test Passes --> O[Output];
```

**G3: MapCoder (Cycle)**
```mermaid
graph LR;
    PL(Planner) --> C(Coder);
    C --> R(Reviewer);
    R --> T(Tester);
    T -- Debug Iteration --> PL;
    T -- Success --> O[Output];
```

**G4: Parallel Judge**
```mermaid
graph TD;
    PL(Planner) --> C(Coder);
    C --> J(Judge);
    C --> T(Tester);
    J --> O[Output];
    T --> O[Output];
```

**G5: Adversarial Debate**
```mermaid
graph LR;
    PL(Planner) --> C(Coder);
    C --> RT(Red Team);
    C --> BT(Blue Team);
    RT <-. Debate .-> BT;
    RT --> T(Tester);
    BT --> T(Tester);
```

**G6: Hierarchical Setup**
```mermaid
graph TD;
    M(Manager) -->|Assigns subtasks| W_Node(Worker Agent)
    
    subgraph Logical Execution
    W_Node -.-> W1(Logical Worker 1)
    W_Node -.-> W2(Logical Worker 2)
    W_Node -.-> W3(...)
    end
    
    W1 -.-> A(Aggregator)
    W2 -.-> A(Aggregator)
    W3 -.-> A(Aggregator)
    A --> T(Tester)
```
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code_acc),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code_cost),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code_nash),
    nbf.v4.new_markdown_cell(text_graphs)
]

with open('analyze_results.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created analyze_results.ipynb successfully!")
