from games import Game
from enum import Enum

# Every game must end in one of three outcomes: 
# either our AI agent wins, loses, or draws against the opponent.
Outcome = Enum("Outcome", ["WIN", "LOSS", "DRAW"])

# A utility function that simply prints out the current state of the game.         
def print_game_object(game: Game):
    print(game)

# Normalizer that places input value somewhere in [0,1]
def get_normalized_value(input_value, min_value, max_value):
    assert max_value != min_value
    return (input_value - min_value) / (max_value - min_value)