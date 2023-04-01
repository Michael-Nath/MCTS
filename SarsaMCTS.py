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
"""

class SarsaMCTS(Player):
    def __init__(self, game: Game, mark, opponent_mark, playout_policy: Policy, exploration_constant=1):
        """
        Initializes the Sarsa MCTS algorithm with a game for it to play.

        Args:
        game (Game): The `Game` object that the MCTS agent interfaces with.
        mark (int): The int representation of the mark the MCTS agent can use to make moves.
        opponent_mark (int): The int representation of the mark the opponent can use to make moves.
        playout_policy: (Policy): The policy that the MCTS agent will follow when at a new state.
        """

        self.game_obj = game
        init_state = game.get_init_game_state()
        # We begin with the initial state of the game we're playing
        self.root = SarsaNode(init_state)
        self.mark = mark
        self.discount_factor = 0.9
        self.alpha = 0.01
        self.V_init: Callable[[SarsaNode], int] = lambda _ : 0
        # Keep track of worst and best returns for normalization downstream.
        self.worst_return = 1e9
        self.best_return = -1e9 
        self.V_playout = self.V_init
        self.trace_decay = 0.5
        self.opponent_mark = opponent_mark
        self.playout_policy = playout_policy
        self.exploration_constant = exploration_constant

        # Maintain an "experience" of previously played game trees. Equivalent to MCTS "learning".
        self.memory: dict[str: SarsaNode] = dict()

    
    
    def ucb1_tree_policy_(self, game_state: SarsaNode) -> np.ndarray:
        best_a = None
        best_ucb = -np.inf
        actions = game_state.game_obj.get_all_next_actions()
        for a_i in actions:
            if game_state.children_states.get(a_i, None) != None:
               child = game_state.children_states[a_i]
               exploit_value = get_normalized_value(child.V, self.worst_return, self.best_return)
               explore_bonus = self.exploration_constant \
                   * np.sqrt(2 * np.log(game_state.n_visited) / np.log(child.n_visited))
               ucb_value = exploit_value + explore_bonus
            else:
               ucb_value = np.inf
            if ucb_value >= best_ucb:
                best_ucb = ucb_value
                best_a = a_i
        return best_a
    
    def generate_episode_(self, root_node: SarsaNode):
        episode = []
        s = root_node.game_obj
        is_opponent_turn = root_node.is_opponent_turn
        while (not self.game_obj.is_terminal_state(s)[0]):
            if self.memory.get(s, None) != None:
                a = self.ucb1_tree_policy_(s) 
            else:
                a = self.playout_policy.select_action(s.state) # playout phase
            sp: Game = s.get_next_game_state(a, self.mark if is_opponent_turn else self.opponent_mark)
            is_opponent_turn = not is_opponent_turn
            is_terminal, winner = self.game_obj.is_terminal_state(sp)
            if is_terminal:
                if winner == self.mark:
                    r = 1
                elif winner == self.opponent_mark:
                    r = -1
            else:
                r = 0
            # EDGE CASE: We append a "throw-away" transition so that root node is backed up
            # for its root-to-next-state transition
            if len(episode) == 0:
                episode.append((None, None, 0, s))
            episode.append((s, a, r, sp)) # s a r s' a' (well, almost)
            s = sp
        return episode
    
    def expand_tree_(self, episode: List[Tuple[Game, List[int], int, Game]]):
        for transition in episode:
            s, a, _, sp = transition
            if self.memory.get(str(sp), None) is None:
                # `sp` will be the first state not found in tree, so `s` MUST be in tree
                parent_node = self.memory.get(str(s), None)
                assert parent_node is not None
                init_v = self.V_init(sp)
                parent_node.add_child(sp, init_v, a)
                self.memory[str(sp)] = parent_node.children_states[a] 
                return
    
    def backup_td_errors_(self, episode: List[Tuple[Game, List[int], int, Game]]):
        td_cum = 0
        v_next = 0
        for (_,_,r,sp) in episode[::-1]: # NOTE: we do not perform backup for root which is OPPONENT
            stringified_game_state = str(sp)
            if self.memory.get(stringified_game_state, None) is not None:
                v_current = self.memory[stringified_game_state].V
            else:
                v_current = self.V_playout(sp)
                
            single_step_td = r + self.discount_factor * v_next - v_current
            td_cum = (self.trace_decay * self.discount_factor * td_cum) + single_step_td
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
        # We begin planning by examining the current state of the game. 
        stringified_current_game_state = str(self.game_obj)
        self.root = self.memory.get(stringified_current_game_state, None)
        if self.root is None:
            self.root = self.memory[stringified_current_game_state] = \
            SarsaNode(self.game_obj, v_init=0, input_action=None, is_opponent=True)
        episode = self.generate_episode_(self.root)
        self.expand_tree_(episode)
        self.backup_td_errors_(episode)
    
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