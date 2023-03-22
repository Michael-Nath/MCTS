import numpy as np
class MCTSNode:
    def __init__(self, game_state):
        self.game_state = game_state
        self.value = 0
