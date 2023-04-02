from utils import Outcome
from games.Game import Game
from MCTSNode import MCTSNode

class NaiveNode(MCTSNode):
    def __init__(self, game_state: Game, input_action = None, is_opponent=False):
        super().__init__(game_state, input_action, is_opponent)
        self.n_won = 0
        self.n_visited = 1
        self.children_states: set[NaiveNode] = set()

    def add_child(self, game_obj, input_action, v_init=0):
        new_child = NaiveNode(game_obj, input_action=input_action, is_opponent=(not self.is_opponent_turn))
        self.children_states.add(new_child)
    
    def add_children(self, game_objs, input_actions):
        for idx, child in enumerate(game_objs):
            self.add_child(child, input_actions[idx]) 
    
    def get_value(self):
        return self.n_won / self.n_visited
    
    def update_stats(self, outcome: Outcome):
        if outcome == outcome.WIN:
            if not self.is_opponent_turn:
                self.n_won += 1
        elif outcome == outcome.LOSS:
            if self.is_opponent_turn:
                self.n_won += 1
        else:
            self.n_won += 0.5
        self.n_visited += 1    
    
    def is_leaf(self):
        return len(self.children_states) == 0

    def __str__(self):
        stringified_state = str(self.game_state)
        return f"s = {stringified_state} | W(s) = {self.n_won} | N(s) = {self.n_visited}"
