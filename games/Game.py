"""
File: Game.py
Author: Michael Nath

This file houses an abstract class called `Game` that describes methods every game must be able
to provide. For instance, it must be possible to get the next reachable states of the game when asked.
These methods will be called by the MCTS agent (e.g. for the above example, the agent may need
the next states during the expansion stage). 
"""
import numpy as np
from typing import Tuple, Union, List
from games.Player import Player

class Game:
    def __init__(self, state):
        self.state = state
        
    def __str__(self):
        raise NotImplementedError    
    
    def copy_(self):
        """
        Internal function that creates a deep copy of the game object instance.
        """
        raise NotImplementedError
        
    @staticmethod
    def get_init_game_state():
        """
        Return the initial game state (no actions taken from either party).
        """
        raise NotImplementedError

    @staticmethod
    def is_terminal_state(game_obj: 'Game') -> Tuple[bool, Union[str, None]]: 
        """
        Function that determines if the current game object is in a terminal (equivalently, 'deciding') 
        state.
        
        Args:
        game_obj (`Game`): The game object to be examined for terminality.
        
        Returns:
        
        is_terminal (bool): A boolean indicating whether this indeed is a terminal state or not.
        winner (str | None): If state terminal, this is the mark of the winner (or None if tie.)
        """
        raise NotImplementedError
    @staticmethod
    def get_reward(game_obj: 'Game', player: Player) -> int:
        """
        The quintessential RL reward function. Takes in the current state, and spits out the reward.
        Note that this not a function of the action leading to the current state. This is due to the
        SarsaMCTS agent treating nodes as afterstates, in which only the value function 
        (function of only the state) is maintained.
        
        Args:
        game_obj (`Game`): The game object instance whose state we we desire the reward of
        player (`Player`): The player responsible for the current state of `game_obj`, influences reward
        if state is terminal.
        """
        raise NotImplementedError
    def get_next_game_states(self, mark: str) -> Tuple[List['Game'], List[int]]:
        """
        Function that presents all next states that can be reached from the current state and if
        marked with `mark`.
        
        Args:
        mark (str): The additional mark that will be in all next states.
        
        Returns:
        (pos_states, input_actions) (Tuple[List[Game], List[int]]): the next states along with the
        corresponding actions that will get the current state to those states.
        """
        raise NotImplementedError
    
    def get_next_game_state(self, action: np.ndarray, mark: str):
        """
        Function that presents the next states that is reached from being at the current state and
        taking `action` with mark `mark`.
        
        Args:
        action (np.ndarray): The action being inputted to the current state.
        mark (str): The additional mark that will be in all next states.
        
        Returns:
        (next_state) (`Game`): The resulting next state.
        """
        raise NotImplementedError
    
    def get_all_next_actions(self) -> List[List[int]]:
        """
        Gets all actions that can be taken from the current state. Whereas `get_all_next_game_states`
        includes the resulting states, this function only includes the input actions.
        """
        
        raise NotImplementedError
    def get_current_game_state(self) -> np.ndarray:
        """
        Provides the current, internal state of the game object.
        """
        raise NotImplementedError

