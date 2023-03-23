import numpy as np
class MCTSNode:
    def __init__(self, game_state):
        self.game_state = game_state
        self.value = 0
        self.children_states = []

    def add_possible_next_state(self, state):
        self.children_states.append(state)
