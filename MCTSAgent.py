from games.Player import Player
from games.Game import Game
from MCTSNode import MCTSNode
from policies.Policy import Policy
from typing import Optional, Any
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
                 exploration_constant: int,
                 alpha: Optional[float] = None,
                 gamma: Optional[float] = None,
                 trace_decay: Optional[float] = None,
                 ):
        # Equipping MCTS Agent with necessary setup regarding the game it will be playing
        self.mark = mark
        super().__init__(self.mark)
        self.opponent_mark = opponent_mark
        self.game_obj = game
        # Configuring internals of MCTS game tree
        self.init_state = game.get_init_game_state()
        # Initializing Hyperparameters
        self.exploration_constant = exploration_constant
        self.alpha = alpha
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.playout_policy = playout_policy
        # Maintain an "experience" of previously played game trees. Equivalent to MCTS "learning".
        # This represents the memorized part of the state space. 
        self.memory: dict[str: MCTSNode] = dict()
    
    
    def pre_step_setup_(self):
        """
        Internal function that gets the MCTSAgent set up before it steps.
        """
        raise NotImplementedError
     
    def step(self) -> None:
        """
        Performs a single iteration of the MCTS algorithm by executing the selection, expansion, simulation, and backpropagation steps. 
        This method should be implemented by subclasses to follow the specific MCTS algorithm's design, leveraging the other methods provided in the subclass.
        
        This is equivalent to a human player `thinking` about what move to make, given their opponent's most recent move. 
        Here, the core assumption is that this is called right after an opponent has made a move. 
        """
        raise NotImplementedError
    
    # NOTE: It is completely up to the specific variant to introduce whatever args/kwargs to 
    # the method signature, as long the variant achieves the purpose of each method.
    def selection_(self, *args, **kwargs) -> Any:
        """
        Performs the selection step of the MCTS algorithm by traversing the game tree according to a tree policy. 
        This method should be implemented by subclasses, specifying the tree policy and additional arguments or keyword arguments as needed. 
        The goal of the selection step is to choose the most promising node to expand.
        """
        raise NotImplementedError
    def expansion_(self, *args, **kwargs) -> Any:
        """
        Performs the expansion step of the MCTS algorithm by generating all possible child nodes from a given leaf node. 
        This method should be implemented by subclasses, considering the specific MCTS algorithm's requirements 
        and additional arguments or keyword arguments as needed. The expansion step helps explore new game states and potential moves.
        """
        raise NotImplementedError
    def simulation_(self, *args, **kwargs) -> Any:
        """
        Performs the simulation step of the MCTS algorithm by executing a rollout (also known as a playout) 
        from the expanded node to a terminal state using a simulation policy. This method should be 
        implemented by subclasses, specifying the simulation policy and additional arguments or keyword arguments as needed. 
        The simulation step provides an estimate of the value for the expanded node based on the outcome of the playout.
        """
        raise NotImplemented
    def backpropagation_(self, *args, **kwargs) -> Any:
        """
        Performs the backpropagation step of the MCTS algorithm by updating the value estimates for each visited node, 
        propagating the outcome of the simulation from the terminal state back to the root node. 
        This method should be implemented by subclasses, considering the specific MCTS algorithm's requirements and 
        additional arguments or keyword arguments as needed. 
        Backpropagation enables the agent to learn from the simulation results and refine its decision-making process.
        """
        raise NotImplemented
    