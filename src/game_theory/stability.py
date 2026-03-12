from typing import Dict

def is_nash_equilibrium(
    base_utilities: Dict[str, float],
    deviation_utilities: Dict[str, float],
    epsilon: float = 0.05
) -> Dict[str, bool]:
    """
    Checks the approximate Nash Equilibrium condition for each agent.
    U_i(e_i=1, e_{-i}^*) >= U_i(e_i=0, e_{-i}^*) - epsilon
    
    Args:
        base_utilities: U_i when all agents perform high effort (e=1).
        deviation_utilities: U_i when agent i unilaterally deviates to low effort (e_i=0).
        epsilon: The tolerance factor.
        
    Returns:
        Dict mapping agent roles to a boolean indicating if they are incentivized 
        to maintain high effort (True = stable, False = incentivized to deviate).
    """
    stability = {}
    for agent, base_u in base_utilities.items():
        if agent in deviation_utilities:
            dev_u = deviation_utilities[agent]
            # Condition: Base utility is at least the deviated utility minus epsilon
            stability[agent] = base_u >= (dev_u - epsilon)
        else:
            # If we didn't test deviation for this agent, assume stable or undefined.
            stability[agent] = True
            
    return stability
