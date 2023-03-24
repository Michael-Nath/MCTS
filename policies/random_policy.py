from games.tictactoe import TicTacToe
import numpy as np
import random

class RandomTTTPolicy:
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
