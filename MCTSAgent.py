from games.Player import Player
from games.Game import Game
from policies.Policy import Policy
class MCTSAgent(Player):
    """
    MCTSAgent: Abstract Base Class for Monte Carlo Tree Search (MCTS) Agents

    This abstract class provides the base functionality for Monte Carlo Tree Search (MCTS) agents. 
    Subclasses, such as SarsaMCTS and NaiveMCTS, can be created to implement specific MCTS algorithms. 
    MCTSAgent assumes that a `Game` object is provided to interact with the environment and that the agent has a specific mark to represent its moves.

    MCTS algorithms generally consist of four stages:

    `Selection`: Starting from the root node, the agent traverses the tree according to a tree policy (e.g., UCB1) until it reaches a leaf node.
    `Expansion`: The agent expands the leaf node by generating all possible child nodes from the current state.
    `Simulation`: The agent performs a rollout (also known as a playout) from the expanded node to a terminal state using a simulation policy, which can be random or follow specific rules.
    `Backpropagation`: The agent updates the value estimates for each visited node by propagating the outcome of the simulation from the terminal state back to the root node.
    This abstract class should be subclassed by specific MCTS agents, which need to implement their own tree policy, expansion method, simulation policy, and backpropagation method.
    """
    def __init__(self,
                 game: Game,
                 mark: str,
                 opponent_mark: str,
                 playout_policy: Policy,
                 
                 )