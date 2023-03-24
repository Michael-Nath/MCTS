import numpy as np
class MCTSNode:
    def __init__(self, game_state):
        self.game_state = game_state
        self.value = 0
        self.children_states = set()

    def add_child(self, state):
        new_child = MCTSNode(state)
        self.children_states.add(new_child)
    
    def add_children(self, states):
        for child in states:
            self.add_child(child)
    
    def is_leaf(self):
        return len(self.children_states) == 0
