import numpy as np
from utils import Outcome

class MCTSNode:
    def __init__(self, game_state, input_action = None, is_opponent=False):
        self.game_state = game_state
        self.n_won = 0
        self.n_visited = 1
        self.input_action = input_action
        self.is_opponent_turn = is_opponent
        self.children_states: set[MCTSNode] = set()

    def add_child(self, state, input_action):
        new_child = MCTSNode(state, input_action=input_action, is_opponent=(not self.is_opponent_turn))
        self.children_states.add(new_child)
    
    def add_children(self, states, input_actions):
        for idx, child in enumerate(states):
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
        stringified_state = np.array2string(self.game_state.flatten())
        return f"s = {stringified_state} | W(s) = {self.n_won} | N(s) = {self.n_visited}"
