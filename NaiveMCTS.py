import numpy as np
from MCTSNode import MCTSNode
from games.Game import Game
from games.Player import Player
from collections import deque
from policies.Policy import Policy
from utils import Outcome
import random 

"""
file: NaiveMCTS.py
Author: Michael D. Nath

NaiveMCTS is the MCTS algorithm as described in its Wikipedia page. 
"""

class NaiveMCTS(Player):
    def __init__(self, game: Game, mark, opponent_mark, playout_policy: Policy, exploration_constant=1):
        """
        Initializes the Naive MCTS algorithm with a game for it to play.

        Args:
        game (Game): The `Game` object that the MCTS agent interfaces with.
        mark (int): The int representation of the mark the MCTS agent can use to make moves.
        opponent_mark (int): The int representation of the mark the opponent can use to make moves.
        playout_policy: (Policy): The policy that the MCTS agent will follow when at a new state.
        """

        self.game_obj = game
        init_state = game.get_init_game_state()
        # We begin with the initial state of the game we're playing
        self.root = MCTSNode(init_state)
        self.mark = mark
        self.path: deque[MCTSNode] = deque([])
        self.opponent_mark = opponent_mark
        self.playout_policy = playout_policy
        self.exploration_constant = exploration_constant

        # Maintain an "experience" of previously played game trees. Equivalent to MCTS "learning".
        self.memory: dict[str: MCTSNode] = dict()

    def perform_lookahead(self, root):
        '''
        This is the selection part of the tree search. Given a root node
        representing the current game state, carve out a path through the game
        tree following the UCB1 heuristic.  
        '''
         
        self.path.append(root)
        # Stop search if we are at a leaf node.
        if (root.is_leaf()):
            return
        # For now we will use the UCB1 heuristic.
        C = self.exploration_constant
        most_promising_node = None
        best_value = 0
        for child in root.children_states:
            # We wish to involve the statistic relevant to the MCTS agent. 
            if not child.is_opponent_turn:
                exploitation_value = child.n_won / child.n_visited
            else:
                exploitation_value = 1 - (child.n_won / child.n_visited)
            exploration_bonus = C * np.sqrt(np.log(root.n_visited) / child.n_visited)
            if (exploitation_value + exploration_bonus) >= best_value:
                best_value = exploitation_value + exploration_bonus
                most_promising_node = child 
        self.perform_lookahead(most_promising_node) 

    def create_children_for_node(self, node: MCTSNode):
        # get all possible next states
        possible_next_states, input_actions = \
        self.game_obj.get_next_game_states(self.mark)
        node.add_children(possible_next_states, input_actions) 
    
    def determine_playout_node(self, parent_node: MCTSNode) -> MCTSNode:
        # For now we will just pick uniformly among the children of the former leaf node.
        return random.choice(list(parent_node.children_states))
    
    def perform_playout(self, playout_node: MCTSNode) -> Outcome:
        simulated_game_obj = playout_node.game_obj.copy_()
        opponent_turn = playout_node.is_opponent_turn
        while self.game_obj.is_terminal_state(simulated_game_obj)[0] == False:
            # simulate moves (for both MCTS and opponent) according to specified policy
            row, col = self.playout_policy.select_action(simulated_game_obj.board)
            simulated_game_obj.state[row, col] = self.opponent_mark if opponent_turn else self.mark
            opponent_turn = not opponent_turn
        winner = simulated_game_obj.is_terminal_state(simulated_game_obj)[1]
        if winner == self.mark:
            return Outcome.WIN
        elif winner == self.opponent_mark:
            return Outcome.LOSS
        return Outcome.DRAW
    
    def backpropagate_outcome(self, outcome: Outcome):
        # for each node in the path from root to non-terminal leaf, update its stored statistics.
        for node in self.path:
            node.update_stats(outcome) 
    
    def step(self):
        '''
        This is equivalent to a human player `thinking` about what move to make,
        given their opponent's most recent move. Here, the core assumption is that
        this is called right after an opponent has made a move. 
        '''
       
        # Edge case: if current game state is already deciding, no point in planning.
        if self.game_obj.is_terminal_state(self.game_obj)[0]:
            return 
        
        # Flush out old path to prepare for next iteration of step().
        self.path = deque([])

        # We begin planning by examining the current state of the game. 
        stringified_current_game_state = np.array2string(self.game_obj.get_current_game_state())
        self.root = self.memory.get(stringified_current_game_state, None)
        if self.root is None:
            self.root = self.memory[stringified_current_game_state] = \
            MCTSNode(self.game_obj, None, is_opponent=True)

        self.perform_lookahead(self.root)
        # At this point, self.path should be populated with a carve-out of game tree.
        leaf_node = self.path[-1]
        # We will construct the next game state from the terminal game state
        # determine if root is terminal (game state is deciding)
        is_terminal, winner = self.game_obj.is_terminal_state(leaf_node.game_obj)
        if is_terminal:
            return
        else:
            self.create_children_for_node(leaf_node)
            playout_node = self.determine_playout_node(leaf_node)
            # Include this playout node as an additional target of backpropagation.
            self.path.append(playout_node)
            # Now we perform a playout from this playout node, backpropagating after playout completion.
            outcome = self.perform_playout(playout_node)
        # Update internal statistics of all nodes in carved out path.
        self.backpropagate_outcome(outcome)
    
    def make_move(self):
        # Perform a one-step lookahead and greedily choose the move to take.
        max_value = 0
        best_child = None
        for child in self.root.children_states:
            if (child.get_value() >= max_value):
                max_value = child.get_value()
                best_child = child
        return best_child.input_action
              
    def internal_print_game_tree_(self, root: MCTSNode):
        if self.game_obj.is_terminal_state(root.game_obj)[0]:
            return
        print(root)
        for child in root.children_states:
            self.internal_print_game_tree_(child)
        
    def print_game_tree(self):
        self.internal_print_game_tree_(self.root)    
        
    def __str__(self):
        return self.root.__str__()