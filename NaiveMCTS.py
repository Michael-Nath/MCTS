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
        """
        Initializes the Naive MCTS algorithm with a game for it to play.

        Args:
        game (Game): This is a Game object that MCTS algorithm interfaces with.
        """
        self.game_obj = game
        init_state = game.get_init_game_state()
        # We begin with the initial state of the game we're playing
        self.root = MCTSNode(init_state)

    # A utility function that simply prints out the current state of the game.         
    # TODO: move this to some utils file.
    def print_game_object(self):
        print(self.game_obj)
    
    def perform_lookahead(self, root):
        '''
        This is the selection part of the tree search. Given a root node
        representing the current game state, carve out a path through the game
        tree following the UCB1 heuristic.  
        '''
         
        # determine if root is terminal (game state is deciding)
        is_terminal, winner = self.game_obj.is_terminal_state(root.game_state)
        if is_terminal:
            print(f"The game state right now is terminal with winner {winner}.")
            return


    def step(self):
        '''
        This is equivalent to a human player `thinking` about what move to make,
        given their opponent's most recent move. Here, the core assumption is that
        this is called right after an opponent has made a move. 
        '''

        # We begin planning by examining the current state of the game. 
        self.root = MCTSNode(self.game_obj.get_current_game_state())
        self.perform_lookahead(self.root)
