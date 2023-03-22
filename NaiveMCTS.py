import numpy as np
from trees.MCTSNode import MCTSNode
from games.tictactoe.Game import Game
"""
file: NaiveMCTS.py
Author: Michael D. Nath

NaiveMCTS is the MCTS algorithm as described in its Wikipedia page. 
"""

class NaiveMCTS():
    def __init__(self, game: Game):
        self.game_obj = game
        init_state = game.get_init_game_state()
        self.root = MCTSNode(init_state)

    def print_game_object(self):
        print(self.game_obj)
    
    def perform_lookahead(self, root):
        '''
        This is the selection part of the tree search. Given a root node
        representing the current game state, carve out a path through the game
        tree following the UCB1 heuristic.  
        '''
        
        # determine if root is terminal (game state is deciding)
        if self.game_obj.is_terminal_state(root):
            return


    def step(self):
        self.root = self.game_obj.get_current_game_state()
        perform_lookahead()
