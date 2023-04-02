import numpy as np
from agents.SarsaNode import SarsaNode
from games.Game import Game
from agents.MCTSAgent import MCTSAgent
from policies.Policy import Policy
from utils import get_normalized_value
from typing import List, Tuple, Callable

class SarsaMCTS(MCTSAgent):
    """
    `SarsaMCTS` is a temporal difference (TD) powered variant of the Monte Carlo Tree Seach (MCTS) algorithm.
    The algorithim below is a high-fidelity implementation of Sarsa-UCT(\lambda) 
    as described by Vodopivec et. al. in "On Monte Carlo Tree Search and Reinforcement Learning".

    `SarsaMCTS` is a class that combines the strengths of the Monte Carlo Tree Search (MCTS) algorithm and SARSA, 
    a temporal difference (TD) learning method, to create a powerful reinforcement learning agent for playing games. 
    The algorithm includes eligibility traces to improve the learning process and maintain the trajectory of the agent throughout the game.
    The class `SarsaMCTS` inherits from `MCTSAgent`, taking several parameters, 
    including the game object, player marks, a playout policy, exploration constant, learning rate, discount factor, and trace decay.

    Key methods in the SarsaMCTS class:

    `ucb1_tree_policy_`: This method calculates the Upper Confidence Bound (UCB1) heuristic to guide the tree traversal and selects the next best action based on the exploration-exploitation trade-off.
    `generate_episode_`: This method generates an episode (trajectory) by following either the tree policy (if the state is memorized) or the playout policy (if the state is new). This approach allows the agent to benefit from both MCTS playouts and reward-seeking frameworks.
    `expand_tree_`: This method expands the game tree based on the current trajectory, updating the memory with the new game state as a child of its predecessor state. It efficiently adapts the representation policy of the agent.
    `backup_td_errors_`: This method performs MCTS backpropagation using the TD learning approach, leveraging eligibility traces to average all possible n-step returns from a state. This method updates the values of all states participating in the trajectory and adjusts the best and worst returns for normalization.
    `step`: This public method performs one iteration of the SarsaMCTS search. It's equivalent to the agent thinking about its next move after an opponent's move. It handles the selection/simulation, expansion, and backpropagation phases.
    `make_move`: This public method returns the most promising move for the agent based on its search.
    `print_game_tree`: This public method prints out the game tree for debugging purposes.

    By leveraging the strengths of MCTS and SARSA, the SarsaMCTS class provides a robust and flexible reinforcement learning agent that can efficiently adapt and learn in a variety of game environments.

    Since SarsaMCTS is a reinforcement learning + monte carlo tree search algorithm, 
    you may see ideas such as episode trajectories interwoven with monte carlo playouts.
    I have attempted to distinguish the RL parts and the search parts, but this algorithm is best 
    appreciated when piecing these components together.
    """
    def __init__(self, 
                 game: Game, 
                 mark, 
                 opponent_mark,
                 playout_policy: Policy, 
                 exploration_constant: float =1,
                 alpha: float = 0.01,
                 gamma: float = 0.9,
                 trace_decay: float = 0.5
                 ):
        """
        Initializes the Sarsa MCTS algorithm with a game for it to play.

        Args:
        game (Game): The `Game` object that the MCTS agent interfaces with.
        mark (int): The int representation of the mark the MCTS agent can use to make moves.
        opponent_mark (int): The int representation of the mark the opponent can use to make moves.
        playout_policy (Policy): The policy that the MCTS agent will follow when at a new state.
        exploration_constant (float): The hyperparamter of the UCB1 heuristic,\
                                            intuitively controls the degree of exploration.
        alpha (float): The learning rate that controls how sensitive nodes are to the TD update.
        gamma (float): The canonical RL discount factor that controls myopicness of MCTS agent.
        trace_decay (float): The eligibility trace parameter controlling how rapidly credit to past \
                             nodes should decay.
        """

        
        # Capture the arguments fed to this class so that variables can be init by underlying ABC.
        saved_args = locals()
        del saved_args['self']
        del saved_args['__class__']
        super().__init__(**saved_args)
        self.root = SarsaNode(self.init_state)
        self.V_init: Callable[[SarsaNode], int] = lambda _ : 0 
        self.V_playout = self.V_init

        # Keep track of worst and best returns for normalization downstream.
        self.worst_return = 1e9
        self.best_return = -1e9 
        
        # Store the trajectory the MCTS agent takes in the game environment.
        # As this is also a SARSA agent, the trajectories take the namesake (s,a,r,s',a') form
        # But, a' is omitted since it is never needed for value computation (TD backwards mechanism). 
        self.episode: List[Tuple[Game, List[int], int, Game]] = []
    
    def ucb1_tree_policy_(self, game_state: SarsaNode) -> np.ndarray:
        """
        Internal function that selects the next action (or equivalently, the next state to move to)
        to take according to the UCB1 heuristic proposed by Kocsis & Szepesvári (2006) for game trees.

        Args:
        game_state (`SarsaNode`): The parent tree node from which we wish to select the next best action.
        
        Returns:
        best_a (np.ndarray): The next best action, represented as a numpy array.
        """
        
        best_a = None
        best_ucb = -np.inf
        # best_a is guaranteed to be non-null because ucb1_tree_policy would only be called on 
        # non-terminal states.
        actions = game_state.game_obj.get_all_next_actions()
        for a_i in actions:
            if game_state.children_states.get(a_i, None) is not None:
               child = game_state.children_states[a_i]
               # To avoid any numerical explosions/implosions that might mess with the heuristic, 
               # and as a best practice, we normalize. 
               exploit_value = get_normalized_value(child.V, self.worst_return, self.best_return)
               # NOTE: Divide by zero runtime error is guaranteed to not occur because
               # `child.n_visited` will be >= 1 by execution of this line (refer to `expand_tree_()`)
               explore_bonus = self.exploration_constant \
                   * np.sqrt(2 * np.log(game_state.n_visited) / np.log(child.n_visited))
               ucb_value = exploit_value + explore_bonus
            # This guarantees the agent sweeps across all possible actions first. 
            else:
               ucb_value = np.inf
            if ucb_value >= best_ucb:
                best_ucb = ucb_value
                best_a = a_i
        return best_a
    def generate_episode_(self, root_node: SarsaNode):
        """
        Internal function that generates a trajectory upon following either a playout policy or 
        tree policy, the choice depending on whether the agent arrives at a memorized state or not.
        This effectively endows the SarsaMCTS agent the benefit of playouts (MCTS feature) under a 
        reward-seeking framework (RL theory).
        For instance, Naive MCTS methods delay reward propagation until at terminal state.

        Args:
        root_node (`SarsaNode`): The tree node (s_0 in RL language) from which to simulate a trajectory.
        
        Side Effects:
        `self.episode` is populated with the trajectory information by end of function execution.
        
        Returns:
        None
        """
        s = root_node.game_obj
        is_opponent_turn = root_node.is_opponent_turn
        while (not self.game_obj.is_terminal_state(s)[0]):
            # This state is memorized, invoke MCTS tree policy
            if self.memory.get(s, None) != None:
                a = self.ucb1_tree_policy_(s) 
            else: # This state is NOT memorized, invoke playout policy (also MCTS theory).
                a = self.playout_policy.select_action(s.state) # playout phase
            sp: Game = s.get_next_game_state(a, self.mark if is_opponent_turn else self.opponent_mark)
            is_opponent_turn = not is_opponent_turn
            # RL theory: instead of waiting for reward signal at termnial state, we get it as we go.
            r = self.game_obj.get_reward(sp, self) 
            # EDGE CASE: We append a "throw-away" transition so that root node is included in backup
            # for its root-to-next-state transition contribution.
            if len(self.episode) == 0:
                self.episode.append((None, None, 0, s))
            self.episode.append((s, a, r, sp)) # s a r s' a' (well, almost)
            s = sp
    
    def expand_tree_(self):
        """
        Internal function that expands out the game tree based on the current trajectory
        This effectively endows the SarsaMCTS agent an adaptive representation policy, where
        the current policy is to only expand the tree once per trajectory. Note that this policy
        is somewhat memory-efficient since nodes are added once per step (MCTS-specific advantage), 
        but is sample-inefficient (RL-specific disadvantage).
        
        Args:
        None
        
        Side Effects:
        `self.memory` is updated to memorize a new game state (sp in RL theory, child node in MCTS theory). 
        This game state is also added as a child of its respective predecessor state (s in RL theory,
        parent node in MCTS theory).

        Returns:
        None
        """
        for (s, a, _, sp) in self.episode[1:]:
            parent_node = self.memory.get(str(s), None)
            # By this algorithm's construction, `s` is guaranteed have been memorized in game tree.
            assert parent_node is not None
            # Add this state as a child of its predecessor.
            if a not in parent_node.children_states.keys():
                init_v = self.V_init(sp)
                parent_node.add_child(sp, init_v, a) 
            # "Expanding" the tree by including it in the memory buffer.
            if self.memory.get(str(sp), None) is None:
                self.memory[str(sp)] = parent_node.children_states[a] 
                return
    
    def backup_td_errors_(self):
        """
        Internal function that performs the MCTS backpropagation in a offline, TD fashion.
        Eligibility traces are leveraged to average out all possible n-step returns from a given state
        with mathematically convenient weighting properties. TD errors are accumulated as 
        the episode is processed backwards, but the accumulation fades away by a factor of `trace_decay`
        each time. The key intuition is that older states should generally receive less credit than the "game-winning"
        states.
        
        Args:
        None
        
        Side Effects:
        
        All states participating in the trajectory get their values backed up.\n
        `self.best_return` is updated with the best return seen so far (used for normalization).\n
        `self.worst_return` is updated with the worst return seen so far (used for normalization).\n
        
        Returns: 
        None 
        """
        td_cum = 0
        v_next = 0
        # Process the episode backwards to implement accumulation of TD errors.
        for (_,_,r,sp) in self.episode[::-1]:
            stringified_game_state = str(sp)
            if self.memory.get(stringified_game_state, None) is not None:
                v_current = self.memory[stringified_game_state].V
            else:
                # Since our representation policy forbids multiple expansions per episode, we estimate.
                # MCTS theory
                v_current = self.V_playout(sp)
                
            # RL theory - single step TD target is r + self.gamma * v_next
            single_step_td = r + self.gamma * v_next - v_current

            # Eligibility Tracing 
            # Diminish the accumulated TD and add single step TD, which will appear as 2,3,... -step
            # returns to older and older states. 
            td_cum = (self.trace_decay * self.gamma * td_cum) + single_step_td
            # conditional updating a consequence of representation policy.
            if self.memory.get(stringified_game_state, None) is not None:
                node = self.memory[stringified_game_state]
                n = node.n_visited = node.n_visited + 1
                alpha = 1 / n
                # If state is heavily explored, it should become less and less sensitive to updates.
                node.V += alpha * td_cum
                if node.V >= self.best_return:
                    self.best_return = node.V
                if node.V <= self.worst_return:
                    self.worst_return = node.V - 1e-9 # additional term prevents divide by zero issue
            v_next = v_current
   
   
    def selection_(self):
        self.generate_episode_(self.root)

    # Simulation is already taken care of during selection step.
    def simulation_(self):
        return
    
    def expansion_(self):
        self.expand_tree_()
        
    def backpropagation_(self):
        self.backup_td_errors_()
                
    def pre_step_setup_(self):
        stringified_current_game_state = str(self.game_obj)
        self.root = self.memory.get(stringified_current_game_state, None)
        if self.root is None:
            self.root = self.memory[stringified_current_game_state] = \
            SarsaNode(self.game_obj, v_init=0, input_action=None, is_opponent=True) 
        # Flush out old episode trajectory.
        self.episode = []       
          
    def step(self):
        """
        Primary public function that executs one iteration of the SarsaMCTS search. 
        This is equivalent to a human player `thinking` about what move to make,
        given their opponent's most recent move. Here, the core assumption is that
        this is called right after an opponent has made a move.
        
        Args:
        None
        
        Side Effects:
        The tree becomes more and more experienced.

        Returns:
        None
        """
       
        # Edge case: if current game state is already deciding, no point in planning.
        if self.game_obj.is_terminal_state(self.game_obj)[0]:
            return
        self.pre_step_setup_()
        self.selection_()
        self.simulation_()
        self.expansion_()
        self.backpropagation_()
    
    def make_move(self) -> np.ndarray:
        """
        Public function that causes the MCTS agent to pick what it thinks is the most promising move
        to take given what it has learned through search.
        
        Args:
        None
        
        Returns:
        action (np.ndarray): The best action to take.
        """
        # Perform a one-step lookahead and greedily choose the move to take.
        max_value = -np.inf
        best_child = None
        for child in self.root.children_states.values():
            if (child.V >= max_value):
                max_value = child.V
                best_child = child
        action = best_child.input_action
        return action

    def internal_print_game_tree_(self, root: SarsaNode):
        """
        Debugging function that prints out the game tree in a somewhat understandable way?

        Args:
        root (`SarsaNode`): the root node from which to print the corresponding subtree
        
        Side Effects:
        Prints the game tree.    
        
        Returns:
        None
        """
        if self.game_obj.is_terminal_state(root.game_obj)[0]:
            return
        print(root)
        for child in root.children_states.values():
            self.internal_print_game_tree_(child)
        
    def print_game_tree(self):
        """
        Public function to print out the game tree. Refer to internal_print_game_tree_ for 
        under-the-hood goodness.
        """
        self.internal_print_game_tree_(self.root)    
        
    def __str__(self):
        return self.root.__str__()