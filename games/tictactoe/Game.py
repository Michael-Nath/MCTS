import numpy as np

class Game:
    def __init__(self, state):
        self.state = state
    def __str__(self):
        raise NotImplementedError
    def get_init_game_state(self):
        raise NotImplementedError
    def is_terminal_state(self):
        raise NotImplementedError
    def get_next_game_states(self, state, mark):
        raise NotImplementedError
    def get_current_game_state(self):
        raise NotImplementedError

