from typing import Tuple
from MCTSNode import MCTSNode
from games.Game import Game

class SarsaNode(MCTSNode):
    def __init__(self, 
                 game_state: Game, 
                 v_init: int = 0, 
                 input_action = None, 
                 is_opponent=False
                 ) -> None:
        super().__init__(game_state, input_action, is_opponent)
        self.V = v_init
        self.n_visited = 0
        self.children_states: dict[Tuple[int, int]: SarsaNode] = dict()

    def add_child(self, game_obj, v_init, input_action) -> None:
        new_child = SarsaNode(game_obj, input_action=input_action, v_init=v_init, is_opponent=(not self.is_opponent_turn))
        self.children_states[input_action] = new_child
    
    def add_children(self, game_objs, input_actions) -> None:
        for idx, child in enumerate(game_objs):
            self.add_child(child, input_actions[idx]) 
    
    def is_leaf(self) -> bool:
        return len(self.children_states) == 0

    def __str__(self) -> str:
        stringified_state = str(self.game_obj)
        return f"s = {stringified_state}\n | V(s): {self.V}"
