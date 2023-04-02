from NaiveMCTS import NaiveMCTS
from games.tictactoe.TicTacToe import TicTacToeBoard
from games.Player import Player
from policies.RandomPolicy import RandomTTTPolicy

NUM_ROWS = 3
NUM_COLS = 3

def simulate(
    manual_play=False, 
    mcts_mark="X",
    opponent_mark="O",
    n_tree_iters=100,
    verbose=False,
    exploration_constant=1
    ):

    tictactoe_game = TicTacToeBoard()
    mcts_brain = NaiveMCTS(tictactoe_game, mcts_mark, opponent_mark, RandomTTTPolicy(), exploration_constant=exploration_constant)
    bot_player = Player(opponent_mark)
    bot_policy = RandomTTTPolicy()
    mcts_player = Player(mcts_mark)

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

def run_experiments(n_trials=100, verbose=False): 
    n_mcts_wins = 0
    n_opponent_wins = 0
    n_draws = 0
    for _ in range(n_trials):
        winner = simulate(
            manual_play=False,
            mcts_mark="X",
            opponent_mark="O",
            n_tree_iters=10,            
            verbose=verbose,
            exploration_constant=1
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
# simulate(manual_play=False, n_tree_iters=100, verbose=False)