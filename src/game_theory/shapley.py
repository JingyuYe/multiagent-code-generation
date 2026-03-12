import numpy as np
import itertools
from typing import List, Callable, Dict, FrozenSet

def estimate_shapley_values(
    agents: List[str], 
    value_function: Callable[[List[str]], float], 
    num_permutations: int = None
) -> Dict[str, float]:
    """
    Estimates Shapley values representing marginal contributions of each agent context.
    
    Args:
        agents: List of agent roles (e.g. ['planner', 'coder', 'reviewer', 'tester'])
        value_function: A function mapping an active subset of agents to an expected payoff (e.g. pass@1 rate)
        num_permutations: Number of permutations to sample for Monte Carlo approx. If None, computes exactly.
        
    Returns:
        A dictionary mapping each agent role to its computed Shapley value phi_i.
    """
    n = len(agents)
    # If the number of permutations is not provided or fits comfortably, calculate exactly.
    if num_permutations is None or num_permutations >= np.math.factorial(n):
        perms = list(itertools.permutations(agents))
    else:
        # Monte Carlo sampling of permutations pi
        perms = [list(np.random.permutation(agents)) for _ in range(num_permutations)]
        
    shapley_values = {agent: 0.0 for agent in agents}
    
    # Memoize value_function to prevent redundant expensive API/sandbox calls
    cache: Dict[FrozenSet[str], float] = {}
    
    def get_val(subset: frozenset) -> float:
        if subset not in cache:
            # Reconstruct list to pass it to the valuation function
            cache[subset] = value_function(list(subset))
        return cache[subset]
        
    for idx, perm in enumerate(perms):
        current_coalition = set()
        v_prev = get_val(frozenset(current_coalition))
        
        for agent in perm:
            current_coalition.add(agent)
            v_curr = get_val(frozenset(current_coalition))
            
            # Marginal contribution to this specific permutation path
            marginal_contribution = v_curr - v_prev
            shapley_values[agent] += marginal_contribution
            
            v_prev = v_curr
            
    # Normalize by the number of permutations sampled
    for agent in agents:
        shapley_values[agent] /= len(perms)
        
    return shapley_values
