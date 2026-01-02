NUM_CONNECT_FOUR_ROWS = 6
NUM_CONNECT_FOUR_COLS = 7
RED_CHECKER = '🔴'
YELLOW_CHECKER = '🟡'
NO_CHECKER = '⚪'


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
    array_repr = to_array_representation(board)
    revised_rows = [''.join([checker(value) for value in row]) for row in array_repr]
    print('\n'.join(revised_rows))


def create_empty_board():
    return [[] for _ in range(NUM_CONNECT_FOUR_COLS)]


def to_array_representation(board):
    connect_four_board = [[0 for _ in range(NUM_CONNECT_FOUR_COLS)] for _ in range(NUM_CONNECT_FOUR_ROWS)]
    for i, column in enumerate(board):
        for j, value in enumerate(column):
            connect_four_board[NUM_CONNECT_FOUR_ROWS-1-j][i] = value
    return connect_four_board        


def all_windows(board):
    array = to_array_representation(board)
    directions = [(1, 1), (-1, 1), (1, 0), (0, 1)]
    windows = []
    for r in range(NUM_CONNECT_FOUR_ROWS):
        for c in range(NUM_CONNECT_FOUR_COLS):
            for dr, dc in directions:
                if all(
                    0 <= r + i*dr < NUM_CONNECT_FOUR_ROWS and
                    0 <= c + i*dc < NUM_CONNECT_FOUR_COLS
                    for i in range(4)
                ):
                    windows.append(tuple([array[r + i*dr][c + i*dc] for i in range(4)]))
    return windows 


def eval_fn(board, player):
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
   
                
   

def check_win_conditions(board):
    array = to_array_representation(board)
    directions = [(1, 1), (-1, 1), (1, 0), (0, 1)]
    winners = set()
    for player in [1, 2]:
        for r in range(NUM_CONNECT_FOUR_ROWS):
            for c in range(NUM_CONNECT_FOUR_COLS):
                for dr, dc in directions:
                    if all(
                        0 <= r + i*dr < NUM_CONNECT_FOUR_ROWS and
                        0 <= c + i*dc < NUM_CONNECT_FOUR_COLS and
                        array[r + i*dr][c + i*dc] == player
                        for i in range(4)
                    ):
                        winners.add(player)
    return winners


def minimax(board, player, ply, eval_fn):
    if ply == 0:
        return eval_fn(board, player)
    



def add_checker(board, column, player):
    board[column].append(player)






board = create_empty_board()
add_checker(board, 3, 1)
add_checker(board, 3, 2)
add_checker(board, 3, 2)
add_checker(board, 3, 2)
add_checker(board, 3, 1)
add_checker(board, 3, 1)
add_checker(board, 3, 1)
#add_checker(board, 2, 2)
add_checker(board, 4, 1)
add_checker(board, 4, 1)
add_checker(board, 4, 2)
add_checker(board, 5, 1)
add_checker(board, 5, 1)
add_checker(board, 5, 1)
add_checker(board, 5, 2)
print_board(board)
print(check_win_conditions(board))
print(all_windows(board))
