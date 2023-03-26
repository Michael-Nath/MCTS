from games.tictactoe import TicTacToe
import numpy as np
import random
from policies.Policy import Policy

class RandomTTTPolicy(Policy):
    """
    This implements a random tic-tac-toe policy where given a state, 
    the next move is chosen uniformly among the open spots in the game board.
    """
    def __init__(self):
        pass
    def select_action(self, state: np.ndarray):
        # we pick a random coordinate based on what's available.
        avail_rows, avail_cols = np.where(state == -1) 
        avail_coords = list(zip(avail_rows, avail_cols))
        coord = random.choice(avail_coords)
        return coord
        
def main():
    print("Hello world, I am a random policy")
