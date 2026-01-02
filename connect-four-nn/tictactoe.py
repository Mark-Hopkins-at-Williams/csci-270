from search import SearchSpace


X_SQUARE = "❌"
O_SQUARE = "⭕"
EMPTY_SQUARE = "⬜"


def print_board(board):
    def image(board_value):
        if board_value == 0:
            return EMPTY_SQUARE
        elif board_value == 1:
            return X_SQUARE
        elif board_value == 2:
            return O_SQUARE
        else:
            raise Exception(f"Invalid board value (must be 0, 1, or 2): {board_value}")

    revised_rows = ["".join([image(value) for value in row]) for row in board]
    print("\n".join(revised_rows))


def check_win_conditions(board):
    winners = set()
    for row in range(3):
        if board[row][0] > 0 and board[row][0] == board[row][1] == board[row][2]:
            winners.add(board[row][0])
    for col in range(3):
        if board[0][col] > 0 and board[0][col] == board[1][col] == board[2][col]:
            winners.add(board[0][col])
    if board[0][0] > 0 and board[0][0] == board[1][1] == board[2][2]:
        winners.add(board[0][0])
    if board[0][2] > 0 and board[0][2] == board[1][1] == board[2][0]:
        winners.add(board[0][2])
    if len(winners) == 1:
        return list(winners)[0]
    elif len(winners) == 0:
        return 0
    else:
        raise Exception("shouldnt get here!")


def board_full(board):
    total = 0
    for row in board:
        for value in row:
            if value > 0:
                total += 1
    return total == 9


def tic_tac_toe_utility_fn(state, player):
    _, board = state
    winner = check_win_conditions(board)
    other_player = 2 if player == 1 else 1
    if winner == player:
        return 1
    elif winner == other_player:
        return -1
    else:
        return 0


class TicTacToeSearchSpace(SearchSpace):

    def get_start_state(self):
        return (1, [[0 for _ in range(3)] for _ in range(3)])

    def is_final_state(self, state):
        _, board = state
        return check_win_conditions(board) > 0 or board_full(board)

    def get_successors(self, state):
        successors = []
        player, board = state
        other_player = 2 if player == 1 else 1
        for row in range(3):
            for col in range(3):
                if board[row][col] == 0:
                    new_board = [r[:] for r in board]
                    new_board[row][col] = player
                    successors.append(((row, col), (other_player, new_board)))
        return successors


if __name__ == "__main__":
    space = TicTacToeSearchSpace()
    row1 = [1, 0, 0]
    row2 = [0, 1, 0]
    row3 = [0, 0, 2]
    state = (2, [row1, row2, row3])
    print_board(state[1])
    successors = space.get_successors(state)
    for action, (player, board) in successors:
        print(action, player)
        print_board(board)
