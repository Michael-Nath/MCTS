import numpy as np
from NaiveMCTS import NaiveMCTS
from games.tictactoe.TicTacToe import TicTacToeBoard
from games.Player import Player
from policies.RandomPolicy import RandomTTTPolicy
import matplotlib.pyplot as plt

MCTS_INDICATOR = 1
OPP_INDICATOR = 0
N_TRIALS = 10

def simulate(
    manual_play=False, 
    mcts_indicator=1,
    opponent_indicator=0,
    n_tree_iters=10000,
    verbose=False,
    exploration_constant=1/np.sqrt(2)
    ):

    tictactoe_game = TicTacToeBoard()
    mcts_brain = NaiveMCTS(tictactoe_game, mcts_indicator, opponent_indicator, RandomTTTPolicy(), exploration_constant=exploration_constant)
    bot_policy = RandomTTTPolicy()
    bot_player = Player(TicTacToeBoard.indicator_to_mark(opponent_indicator))
    mcts_player = Player(TicTacToeBoard.indicator_to_mark(mcts_indicator))

    while TicTacToeBoard.is_terminal_state(tictactoe_game)[0] == False:
        if manual_play:
            bot_action = input("Provide row, column\n").split(',')
            bot_action = [int(x) for x in bot_action]
        else: 
            bot_action = bot_policy.select_action(tictactoe_game.get_current_game_state())
        tictactoe_game.mark_move(bot_player, int(bot_action[0]), int(bot_action[1]))
        if verbose:
            print(f"Opponent is marking {bot_player.mark} at coordinate {bot_action}")
            print(tictactoe_game)
        if TicTacToeBoard.is_terminal_state(tictactoe_game)[0]:
            break
        for _ in range(n_tree_iters):
            mcts_brain.step()
        mcts_action = mcts_brain.make_move()
        tictactoe_game.mark_move(mcts_player, mcts_action[0], mcts_action[1])
        if verbose:
            print(f"MCTS Agent is marking {mcts_player.mark} at coordinate {mcts_action}")
            print(tictactoe_game)
            
    if verbose:
        print()
        print("TICTACTOE FINAL GAME STATE:")
        print(tictactoe_game)

    _, winner = tictactoe_game.is_terminal_state(tictactoe_game)
    return winner
def run_experiments(
    n_trials=N_TRIALS, 
    verbose=False, 
    exploration_constant=1, 
    n_tree_iters=10000
    ): 
    n_mcts_wins = 0
    n_opponent_wins = 0
    n_draws = 0
    for _ in range(n_trials):
        winner = simulate(
            manual_play=False,
            mcts_indicator=1,
            opponent_indicator=0,
            n_tree_iters=n_tree_iters,            
            verbose=verbose,
            exploration_constant=exploration_constant
        )
        if winner == 1:
            n_mcts_wins += 1
        elif winner == 0:
            n_opponent_wins += 1
        else:
            n_draws += 1
    return n_mcts_wins



def vary_num_tree_iterations(n_max_tree_iters):
    # This determines how granular we want our findings of optimal tree iterations to be.
    n_checkpoints = 10
    n_iterations_set = np.linspace(1, n_max_tree_iters, n_checkpoints, dtype=int)
    mcts_wins = np.array([])
    for n_tree_iters in n_iterations_set:
        mcts_wins = np.append(mcts_wins, run_experiments(n_tree_iters=n_tree_iters))
    plt.plot(n_iterations_set, mcts_wins / 100)
    plt.savefig("plots/n_tree_iters.png")

if __name__ == "__main__":
    vary_num_tree_iterations(10000)