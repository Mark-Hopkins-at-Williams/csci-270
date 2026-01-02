
RED_CHECKER = '❌'
YELLOW_CHECKER = '⭕'
NO_CHECKER = '⬜'

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
    revised_rows = [''.join([checker(value) for value in row]) for row in board]
    print('\n'.join(revised_rows))


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
        raise Exception('shouldnt get here!')



def play(board, player, row, col):
    board[row][col] = player


def board_full(board):
    total = 0
    for row in board:
        for value in row:
            if value > 0:
                total += 1
    return total == 9


def collect_best_moves():

    def minimax(board, whose_turn, who_am_i):
        winner = check_win_conditions(board)
        if winner == who_am_i:
            return 1
        elif winner > 0:
            return -1
        elif board_full(board):
            return 0
        else:
            other_player = 1 if whose_turn == 2 else 2
            minimax_value = float('-inf') if who_am_i == whose_turn else float('inf')
            best_move = []
            for row in range(3):
                for col in range(3):
                    if board[row][col] == 0:
                        board[row][col] = whose_turn
                        if who_am_i == whose_turn:                            
                            next_value = minimax(board, other_player, who_am_i)
                            if next_value > minimax_value:
                                minimax_value = next_value
                                best_move = [(row, col)]
                            elif next_value == minimax_value:
                                best_move.append((row, col))
                        else:
                            next_value = minimax(board, other_player, who_am_i)
                            if next_value < minimax_value:
                                minimax_value = next_value
                                best_move = [(row, col)]
                            elif next_value == minimax_value:
                                best_move.append((row, col))
                        board[row][col] = 0
            board_clone = tuple([tuple(row) for row in board])
            if whose_turn == who_am_i:
                best_moves.append({
                    'board': board_clone,
                    'best_moves': best_move
                })
            return minimax_value
    
    best_moves = []
    board = [[0,0,0],[0,0,0],[0,0,0]]
    minimax(board, 1, 1)
    minimax(board, 1, 2)
    return best_moves
    
    
if __name__ == "__main__":
    import json
    best = collect_best_moves()
    with open('tictactoemoves.json', 'w') as writer:
        json.dump(best, writer)