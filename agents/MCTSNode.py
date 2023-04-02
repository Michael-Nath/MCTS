import numpy as np
from typing import Optional, List
from games.Game import Game
class MCTSNode():
    def __init__(self, 
                 game_state: Game,
                 input_action: np.ndarray = None,
                 is_opponent: bool = False
                 ) -> None:
        self.game_obj = game_state
        self.input_action = input_action
        self.is_opponent_turn = is_opponent

    def add_child(self, game_obj: Game, input_action: List[int], v_init: Optional[int] = 0):
        raise NotImplementedError
    def add_children(self, game_objs: List[Game], input_actions: List[List[int]]):
        raise NotImplementedError
    def is_leaf(self) -> bool:
        raise NotImplementedError
    def __str__(self) -> str:
        raise NotImplementedError
    