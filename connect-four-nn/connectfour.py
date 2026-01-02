from search import SearchSpace
from minimax import create_minimax_function, create_ranked_minimax_function
import random
from tqdm import tqdm
import json

RED_CHECKER = "🔴"
YELLOW_CHECKER = "🟡"
NO_CHECKER = "⚪"


def print_board(board):
    def checker(board_value):
        if board_value == 0:
            return NO_CHECKER
        elif board_value == 1:
            return RED_CHECKER
        elif board_value == 2:
            return YELLOW_CHECKER
        else:
            raise Exception(f"Invalid connect-four board value: {board_value}")

    revised_rows = ["".join([checker(value) for value in row]) for row in board]
    print("\n".join(revised_rows))


def check_win_conditions(board):
    num_rows = len(board)
    num_cols = len(board[0])
    directions = [(1, 1), (-1, 1), (1, 0), (0, 1)]
    winners = set()
    for player in [1, 2]:
        for r in range(num_rows):
            for c in range(num_cols):
                for dr, dc in directions:
                    if all(
                        0 <= r + i * dr < num_rows
                        and 0 <= c + i * dc < num_cols
                        and board[r + i * dr][c + i * dc] == player
                        for i in range(4)
                    ):
                        winners.add(player)
    if len(winners) > 0:
        return list(winners)[0]
    else:
        return 0


def connect_four_utility_fn(state, player):
    _, board = state
    winner = check_win_conditions(board)
    other_player = 2 if player == 1 else 1
    if winner == player:
        return 1
    elif winner == other_player:
        return -1
    else:
        return 0


class ConnectFourSearchSpace(SearchSpace):

    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def get_start_state(self):
        return (1, [[0 for _ in range(self.num_cols)] for _ in range(self.num_rows)])

    def is_final_state(self, state):
        _, board = state
        return check_win_conditions(board) > 0 or len(self.get_successors(state)) == 0

    def get_successors(self, state):
        successors = []
        player, board = state
        other_player = 2 if player == 1 else 1
        for col in range(self.num_cols):
            row = self.num_rows - 1
            while row >= 0 and board[row][col] > 0:
                row -= 1
            if row >= 0:
                new_board = [r[:] for r in board]
                new_board[row][col] = player
                successors.append((col, (other_player, new_board)))
        return successors


def connect_four_evaluation_fn(state, player):
    _, board = state
    windows = all_windows(board)
    opponent = 2 if player == 1 else 1
    total = 0
    for window in windows:
        if window.count(player) == 4:
            total += 100000
        elif window.count(player) == 3 and window.count(0) == 1:
            total += 100
        elif window.count(player) == 2 and window.count(0) == 2:
            total += 10
        elif window.count(opponent) == 3 and window.count(0) == 1:
            total += -120
        elif window.count(opponent) == 2 and window.count(0) == 2:
            total += -10
        elif window.count(opponent) == 4:
            total += -100000
    return total


def all_windows(board):
    directions = [(1, 1), (-1, 1), (1, 0), (0, 1)]
    num_rows, num_cols = len(board), len(board[0])
    windows = []
    for r in range(num_rows):
        for c in range(num_cols):
            for dr, dc in directions:
                if all(
                    0 <= r + i * dr < num_rows and 0 <= c + i * dc < num_cols
                    for i in range(4)
                ):
                    windows.append(
                        tuple([board[r + i * dr][c + i * dc] for i in range(4)])
                    )
    return windows


def create_minimax_ai(player, max_plys, randomization=0.0):
    minimax_fn = create_minimax_function(
        ConnectFourSearchSpace(6, 7),
        connect_four_evaluation_fn,
        player=player,
        max_plys=max_plys,
    )

    def ai_function(current_player, board):
        if random.random() < randomization:
            options = [col for col in range(7) if board[0][col] == 0]
            action = random.choice(options)
        else:
            action, _ = minimax_fn((current_player, board))
        return action

    return ai_function


import sys
import time


def clear_screen():
    # Clear screen and move cursor to (0,0)
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def play_game(player1_ai, player2_ai, visualize=True):
    num_rows, num_cols = 6, 7
    board = [[0] * num_cols for _ in range(num_rows)]
    moves_made = 0
    current_player = 1
    states = []
    while check_win_conditions(board) == 0 and moves_made < 42:
        if visualize:
            clear_screen()
            print_board(board)
            print(f"\nplayer {current_player} is thinking...")
        states.append((current_player, tuple([tuple(r) for r in board])))
        ai = player1_ai if current_player == 1 else player2_ai
        col = ai(current_player, board)
        row = num_rows - 1
        while row >= 0 and board[row][col] > 0:
            row -= 1
        board[row][col] = current_player
        current_player = 2 if current_player == 1 else 1
        moves_made += 1

    winning_player = 2 if current_player == 1 else 1
    if visualize:
        clear_screen()
        print_board(board)
        print(f"\nplayer {winning_player} wins!")
    return states


def create_nn_ai():
    from ml import (
        TwoLayerNet,
        input_dim,
        hidden_dim,
        num_outcomes,
        convert_board_state_to_vector,
    )
    import torch

    model = TwoLayerNet(input_dim, hidden_dim, num_outcomes)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    def ai_function(current_player, board):
        vec = convert_board_state_to_vector(board)
        logits = model(vec)
        pred = torch.argmax(logits, dim=0)
        return pred.item()

    return ai_function


def label_data(max_plys):
    with open("c4boards.json") as reader:
        data = json.load(reader)
    moves = []

    for player, board in tqdm(data[:5]):
        minimax_fn = create_ranked_minimax_function(
            ConnectFourSearchSpace(6, 7),
            connect_four_evaluation_fn,
            player=player,
            max_plys=max_plys,
        )
        evaluated_actions = minimax_fn((player, board))
        print_board(board)
        for item in evaluated_actions:
            print(item)

    with open("c4moves.6ply.json", "w") as writer:
        json.dump(moves, writer)


if __name__ == "__main__":
    label_data(6)

    # ai2 = create_minimax_ai(2, 2, randomization=0.1)
    # ai1 = create_nn_ai()
    # play_game(ai1, ai2)
    # exit()

    # print(connect_four_utility_fn(state, player=1))
    # successors = space.get_successors(state)
    # for action, (player, board) in successors:
    #     print(action, player)
    #     print_board(board)

    # states = set()
    # for player, board in data:
    #     state = (player, tuple([tuple(r) for r in board]))
    #     states.add(state)
    # print(len(states))
    # ai1 = create_minimax_ai(1, 4, randomization=1.0)
    # ai2 = create_minimax_ai(2, 4, randomization=1.0)
    # for _ in tqdm(range(1000)):
    #     for state in play_game(ai1, ai2, visualize=False):
    #         states.add(state)
    # print(len(states))
    # states = list(states)
    # with open("c4boards.json", "w") as writer:
    #     json.dump(states, writer)
