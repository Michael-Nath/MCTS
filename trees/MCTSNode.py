import numpy as np
from utils import Outcome

class MCTSNode:
    def __init__(self, game_state, is_opponent=False):
        self.game_state = game_state
        self.n_won = 0
        self.n_visited = 1
        self.is_opponent_turn = is_opponent
        self.children_states: set[MCTSNode] = set()

    def add_child(self, state):
        new_child = MCTSNode(state, is_opponent=(not self.is_opponent_turn))
        self.children_states.add(new_child)
    
    def add_children(self, states):
        for child in states:
            self.add_child(child) 
    
    def update_stats(self, outcome: Outcome):
        if outcome == outcome.WIN:
            if not self.is_opponent_turn:
                self.n_won += 1
        elif outcome == outcome.DRAW:
            self.n_won += 1
        self.n_visited += 1    
    
    def is_leaf(self):
        return len(self.children_states) == 0

    def __str__(self):
        return np.array2string(self.game_state)