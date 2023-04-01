import numpy as np
from SarsaNode import SarsaNode
from games.Game import Game
from games.Player import Player
from policies.Policy import Policy
from utils import get_normalized_value
from typing import List, Tuple, Callable

"""
file: SarsaMCTS.py
Author: Michael D. Nath

SarsaMCTS is a TD-powered variant of the Monte Carlo Tree Seach algorithm.
In this variant, we leverage the SARSA TD method with eligibility traces.
The algorithim below is a high-fidelity implementation of Sarsa-UCT(\lambda) 
as described by Vodopivec et. al. in "On Monte Carlo Tree Search and Reinforcement Learning".

SarsaMCTS is a reinforcement learning + monte carlo tree search algorithm. As such, 
you may see ideas such as episode trajectories interwoven with monte carlo playouts.
I have attempted to distinguish the RL parts and the search parts, but this algorithm is best 
appreciated when piecing these components together.
"""

class SarsaMCTS(Player):
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

        
        # Equipping with MCTS Agent necessary setup regarding the game it will be playing.
        self.mark = mark 
        self.game_obj = game
        self.opponent_mark = opponent_mark
        
        # Configuring internals of MCTS game tree
        init_state = game.get_init_game_state()
        self.root = SarsaNode(init_state)
        self.V_init: Callable[[SarsaNode], int] = lambda _ : 0 
        self.V_playout = self.V_init

        # Keep track of worst and best returns for normalization downstream.
        self.worst_return = 1e9
        self.best_return = -1e9 

        # Initializing Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.trace_decay = trace_decay
        self.playout_policy = playout_policy
        self.exploration_constant = exploration_constant

        # Maintain an "experience" of previously played game trees. Equivalent to MCTS "learning".
        # This represents the memorized part of the state space. 
        self.memory: dict[str: SarsaNode] = dict()
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
            else: # This state is NOT memorized, invoke playout policy (still MCTS component).
                a = self.playout_policy.select_action(s.state) # playout phase
            sp: Game = s.get_next_game_state(a, self.mark if is_opponent_turn else self.opponent_mark)
            is_opponent_turn = not is_opponent_turn
            
            r = self.game_obj.get_reward(sp)
            
            
            
            is_terminal, winner = self.game_obj.is_terminal_state(sp)
            self. 
            if is_terminal:
                if winner == self.mark:
                    r = 1
                elif winner == self.opponent_mark:
                    r = -1
            else:
                r = 0
            # EDGE CASE: We append a "throw-away" transition so that root node is included in backup
            # for its root-to-next-state transition contribution.
            if len(self.episode) == 0:
                self.episode.append((None, None, 0, s))
            self.episode.append((s, a, r, sp)) # s a r s' a' (well, almost)
            s = sp
    
    def expand_tree_(self):
        for (s, a, _, sp) in self.episode[1:]:
            parent_node = self.memory.get(str(s), None)
            # By this algorithm's construction, `s` will ALWAYS have been memorized in game tree.
            assert parent_node is not None
            if a not in parent_node.children_states.keys():
                init_v = self.V_init(sp)
                parent_node.add_child(sp, init_v, a) 
            if self.memory.get(str(sp), None) is None:
                self.memory[str(sp)] = parent_node.children_states[a] 
                return
    
    def backup_td_errors_(self):
        td_cum = 0
        v_next = 0
        for (_,_,r,sp) in self.episode[::-1]: # NOTE: we do not perform backup for root which is OPPONENT
            stringified_game_state = str(sp)
            if self.memory.get(stringified_game_state, None) is not None:
                v_current = self.memory[stringified_game_state].V
            else:
                v_current = self.V_playout(sp)
                
            single_step_td = r + self.gamma * v_next - v_current
            td_cum = (self.trace_decay * self.gamma * td_cum) + single_step_td
            if self.memory.get(stringified_game_state, None) is not None:
                node = self.memory[stringified_game_state]
                n = node.n_visited = \
                node.n_visited + 1
                alpha = 1 / n
                node.V += alpha * td_cum
                if node.V >= self.best_return:
                    self.best_return = node.V
                if node.V <= self.worst_return:
                    self.worst_return = node.V - 1e-9 # additional term prevents divide by zero issue
            v_next = v_current
    
    def step(self):
        '''
        This is equivalent to a human player `thinking` about what move to make,
        given their opponent's most recent move. Here, the core assumption is that
        this is called right after an opponent has made a move. 
        '''
       
        # Edge case: if current game state is already deciding, no point in planning.
        if self.game_obj.is_terminal_state(self.game_obj)[0]:
            return 
        
        stringified_current_game_state = str(self.game_obj)
        self.root = self.memory.get(stringified_current_game_state, None)
        if self.root is None:
            self.root = self.memory[stringified_current_game_state] = \
            SarsaNode(self.game_obj, v_init=0, input_action=None, is_opponent=True)

        # Flush out old episode trajectory.
        self.episode = []        
        self.generate_episode_(self.root)
        self.expand_tree_()
        self.backup_td_errors_()
    
    def make_move(self):
        # Perform a one-step lookahead and greedily choose the move to take.
        max_value = -np.inf
        best_child = None
        for child in self.root.children_states.values():
            if (child.V >= max_value):
                max_value = child.V
                best_child = child
        return best_child.input_action
              
    def internal_print_game_tree_(self, root: SarsaNode):
        if self.game_obj.is_terminal_state(root.game_obj)[0]:
            return
        print(root)
        for child in root.children_states.values():
            self.internal_print_game_tree_(child)
        
    def print_game_tree(self):
        self.internal_print_game_tree_(self.root)    
        
    def __str__(self):
        return self.root.__str__()