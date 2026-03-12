from typing import Dict

def calculate_utility(
    test_passed: bool, 
    token_usage: Dict[str, int], 
    value_of_success: float = 100.0, 
    lambda_cost: float = 0.01
) -> Dict[str, float]:
    """
    U_i = V * 1[Y=pass] - \lambda * T_i
    
    Args:
        test_passed: execution outcome (Y)
        token_usage: Dictionary mapping agent roles to their total token consumption (T_i)
        value_of_success: Reward for passing tests (V)
        lambda_cost: Scale modifier for cost (\lambda)
        
    Returns:
        Dict mapping agent role to its computed Utility score U_i.
    """
    utilities = {}
    reward = value_of_success if test_passed else 0.0
    
    for agent_role, tokens in token_usage.items():
        cost = lambda_cost * tokens
        utilities[agent_role] = reward - cost
        
    return utilities
