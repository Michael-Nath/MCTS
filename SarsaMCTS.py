import numpy as np
# from MCTSNode import MCTSNode
from SarsaNode import SarsaNode
from games.Game import Game
from games.Player import Player
from collections import deque
from policies.Policy import Policy
from utils import Outcome
import random 

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
        self.discount_factor = 1
        self.alpha = 0.01
        self.trace_decay = 1
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
               exploitation_value = child.get_value()
               exploration_value = self.exploration_constant \
                   * np.sqrt(2 * np.log(game_state.n_visited) / np.log(child.n_visited))
               ucb_value = exploitation_value + exploration_value
               if ucb_value >= best_ucb:
                   best_ucb = ucb_value
                   best_a = a_i
        return best_a
    
    def generate_episode_(self, root_state: SarsaNode):
        episode = []
        s = root_state
        while (not self.game_obj.is_terminal_state(s.game_obj)):
            if self.memory.get(s, None) != None:
                a = self.ucb1_tree_policy_(s) 
            else:
                a = self.playout_policy.select_action(s.game_obj.state)
            sp = self.game_obj.get_next_game_state(a, self.mark)
            is_terminal, winner = self.game_obj.is_terminal_state(sp.game_ob)
            if is_terminal:
                if winner == self.mark:
                    r = 1
                elif winner == self.opponent_mark:
                    r = -1
            else:
                r = 0
            s = sp
            episode.append((s, r))
        
    
    def create_children_for_node(self, node: SarsaNode):
        # get all possible next states
        possible_next_states, input_actions = \
        self.game_obj.get_next_game_states(self.mark)
        node.add_children(possible_next_states, input_actions) 
    
    def backpropagate_outcome(self, outcome: Outcome):
        # for each node in the path from root to non-terminal leaf, update its stored statistics.
        for node in self.path:
            node.update_stats(outcome) 
    
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
        stringified_current_game_state = np.array2string(self.game_obj.get_current_game_state())
        self.root = self.memory.get(stringified_current_game_state, None)
        if self.root is None:
            self.root = self.memory[stringified_current_game_state] = \
            SarsaNode(self.game_obj, None, is_opponent=True)
        episode = self.generate_episode_(self.root)
        
        
        
        
        
        
        self.perform_lookahead(self.root)
        # At this point, self.path should be populated with a carve-out of game tree.
        leaf_node = self.path[-1]
        # We will construct the next game state from the terminal game state
        # determine if root is terminal (game state is deciding)
        is_terminal, winner = self.game_obj.is_terminal_state(leaf_node.game_obj)
        if is_terminal:
            return
        else:
            self.create_children_for_node(leaf_node)
            playout_node = self.determine_playout_node(leaf_node)
            # Include this playout node as an additional target of backpropagation.
            self.path.append(playout_node)
            # Now we perform a playout from this playout node, backpropagating after playout completion.
            outcome = self.perform_playout(playout_node)
        # Update internal statistics of all nodes in carved out path.
        self.backpropagate_outcome(outcome)
    
    def make_move(self):
        # Perform a one-step lookahead and greedily choose the move to take.
        max_value = 0
        best_child = None
        for child in self.root.children_states:
            if (child.get_value() >= max_value):
                max_value = child.get_value()
                best_child = child
        return best_child.input_action
              
    def internal_print_game_tree_(self, root: MCTSNode):
        if self.game_obj.is_terminal_state(root.game_obj)[0]:
            return
        print(root)
        for child in root.children_states:
            self.internal_print_game_tree_(child)
        
    def print_game_tree(self):
        self.internal_print_game_tree_(self.root)    
        
    def __str__(self):
        return self.root.__str__()