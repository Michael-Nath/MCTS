from SarsaMCTS import SarsaMCTS
from games.tictactoe.TicTacToe import TicTacToeBoard
from games.Player import Player
from policies.RandomPolicy import RandomTTTPolicy
import numpy as np

def simulate(
    manual_play=False, 
    mcts_mark="X",
    opponent_mark="O",
    n_tree_iters=10000,
    verbose=False,
    exploration_constant=1/np.sqrt(2)
    ):

    tictactoe_game = TicTacToeBoard()
    mcts_brain = SarsaMCTS(tictactoe_game, 
                           mcts_mark, 
                           opponent_mark, 
                           RandomTTTPolicy(), 
                           exploration_constant=exploration_constant,
                           trace_decay=0.99,
                           gamma=0.90)
    bot_policy = RandomTTTPolicy()
    bot_player = Player(opponent_mark)
    mcts_player = Player(mcts_mark)
    if verbose:
        print("Starting Game Board:\n")
        print(tictactoe_game)
    if manual_play and verbose:
        print(f"You are marking with {bot_player.mark}; MCTS agent is marking with {mcts_player.mark}")
    elif not manual_play and verbose:
        print(f"Bot is marking with {bot_player.mark}; MCTS agent is marking with {mcts_player.mark}")
    while TicTacToeBoard.is_terminal_state(tictactoe_game)[0] == False:
        if manual_play:
            bot_action = input("\nProvide row, column\n").split(',')
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

def run_experiments(n_trials=10, verbose=False): 
    n_mcts_wins = 0
    n_opponent_wins = 0
    n_draws = 0
    for _ in range(n_trials):
        winner = simulate(
            manual_play=False,
            mcts_mark="X",
            opponent_mark="O",
            n_tree_iters=10,
            exploration_constant=1,          
            verbose=verbose,
        )
        if winner == 1:
            n_mcts_wins += 1
        elif winner == 0:
            n_opponent_wins += 1
        else:
            n_draws += 1
    print(f"NUM MCTS WINS: {n_mcts_wins}/{n_trials} = {n_mcts_wins * 100 / n_trials}%") 
    print(f"NUM OPPONENT WINS: {n_opponent_wins}/{n_trials} = {n_opponent_wins * 100 / n_trials}%")
    print(f"NUM DRAWS: {n_draws}/{n_trials} = {n_draws * 100 / n_trials}%")

run_experiments(n_trials=100, verbose=False)
# simulate(manual_play=True, n_tree_iters=1, verbose=True)