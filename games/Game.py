"""
File: Game.py
Author: Michael Nath

This file houses an abstract class called `Game` that describes methods every game must be able
to provide. For instance, it must be possible to get the next reachable states of the game when asked.
These methods will be called by the MCTS agent (e.g. for the above example, the agent may need
the next states during the expansion stage). 

"""
import numpy as np
from typing import Tuple, Union
from games.Player import Player
from multipledispatch import dispatch

class Game:
    def __init__(self, state):
        self.state = state
        self.player_one = None
        self.player_two = None
        
    def __str__(self):
        raise NotImplementedError    
    
    def assign_player_one(self, player: Player):
        self.player_one = player
    
    def assign_player_two(self, player: Player):
        self.player_two = player
    
    def copy_(self):
        raise NotImplementedError
        
    @staticmethod
    def get_init_game_state():
        """
        Return the initial game state (no actions taken from either party).
        """
        raise NotImplementedError

    @staticmethod
    def is_terminal_state(game_obj): 
        """
        
        return:
        is_terminal (bool): A boolean indicating whether this indeed is a terminal state or not.
        winner (Player | None): If state terminal, this is a `Player` object returning the winner 
                                (or None if tie.)
        """
        raise NotImplementedError
    
    def get_next_game_states(self, state, mark):
        raise NotImplementedError
    
    def get_next_game_state(self, action, mark):
        raise NotImplementedError
    
    def get_all_next_actions(self):
        raise NotImplementedError
    def get_current_game_state(self):
        raise NotImplementedError

