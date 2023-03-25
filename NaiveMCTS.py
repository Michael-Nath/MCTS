import numpy as np
from trees.MCTSNode import MCTSNode
from games.tictactoe.Game import Game
from collections import deque
from pprint import pprint
from policies.Policy import Policy
from policies.RandomPolicy import RandomTTTPolicy
from utils import Outcome
"""
file: NaiveMCTS.py
Author: Michael D. Nath

NaiveMCTS is the MCTS algorithm as described in its Wikipedia page. 
"""

class NaiveMCTS():
    def __init__(self, game: Game, mark, opponent_mark, playout_policy: Policy):
        """
        Initializes the Naive MCTS algorithm with a game for it to play.

        Args:
        game (Game): This is a Game object that MCTS algorithm interfaces with.
        """
        self.game_obj = game
        init_state = game.get_init_game_state()
        # We begin with the initial state of the game we're playing
        self.root = MCTSNode(init_state)
        self.mark = mark
        self.path = deque([])
        self.opponent_mark = opponent_mark
        self.playout_policy = playout_policy

    def perform_lookahead(self, root):
        '''
        This is the selection part of the tree search. Given a root node
        representing the current game state, carve out a path through the game
        tree following the UCB1 heuristic.  
        '''
         
        self.path.append(root)
        # Stop search if we are at a leaf node.
        if (root.is_leaf()):
            return
        
    def create_children_for_node(self, node: MCTSNode):
        # get all possible next states
        possible_next_states = self.game_obj.get_next_game_states(node.game_state, self.mark)
        node.add_children(possible_next_states) 
    
    def determine_playout_node(self, parent_node: MCTSNode) -> MCTSNode:
        # For now we will use the UCB1 heuristic.
        C = 1
        playout_node = None
        best_value = 0
        for child in parent_node.children_states:
            exploitation_value = child.n_won / child.n_visited
            exploration_bonus = C * np.sqrt(np.log(parent_node.n_visited) / child.n_visited)
            if (exploitation_value + exploration_bonus) >= best_value:
                best_value = exploitation_value + exploration_bonus
                playout_node = child
        return playout_node

    def perform_playout(self, playout_node: MCTSNode) -> Outcome:
        copy_of_current_game_state = playout_node.game_state.copy()
        opponent_turn = playout_node.is_opponent_turn
        while self.game_obj.is_terminal_state(copy_of_current_game_state)[0] == False:
            # simulate moves (for both MCTS and opponent) according to specified policy
            row, col = self.playout_policy.select_action(copy_of_current_game_state)
            copy_of_current_game_state[row, col] = self.opponent_mark if opponent_turn else self.mark
            opponent_turn = not opponent_turn
        winner = self.game_obj.is_terminal_state(copy_of_current_game_state)[1]
        print(winner)
    def step(self):
        '''
        This is equivalent to a human player `thinking` about what move to make,
        given their opponent's most recent move. Here, the core assumption is that
        this is called right after an opponent has made a move. 
        '''

        # We begin planning by examining the current state of the game. 
        self.root = MCTSNode(self.game_obj.get_current_game_state(), is_opponent=True)
        self.perform_lookahead(self.root)
        # At this point, self.path should be populated with a carve-out of game tree.
        leaf_node = self.path[-1]
        # We will construct the next game state from the terminal game state
        # determine if root is terminal (game state is deciding)
        is_terminal, winner = self.game_obj.is_terminal_state(leaf_node.game_state)
        if is_terminal:
            print(f"The game state right now is terminal with winner {winner}.")
            # return
        self.create_children_for_node(leaf_node)
        playout_node = self.determine_playout_node(leaf_node)
        # Now we perform a playout from this playout node, backpropagating after playout completion.
        self.perform_playout(playout_node)