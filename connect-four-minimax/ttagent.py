import json
from ml import TwoLayerNet, load_training_data, input_dim, hidden_dim
from ml import convert_board_state_to_vector
import torch
from tictactoe import check_win_conditions, print_board, board_full
from random import choice


model = TwoLayerNet(input_dim, hidden_dim)
model.load_state_dict(torch.load("model.pt"))
model.eval()

def load_move_dict():
    with open('tictactoemoves.json') as reader:
        json_data = json.load(reader)
    best_moves = dict()
    for datum in json_data:
        board = []
        for row in datum['board']:
            for element in row:
                board.append(element)
        board = tuple(board)
        moves = []
        for row, col in datum['best_moves']:
            moves.append(row*3 + col)
        best_moves[board] = moves
    return best_moves
    
best_move_dict = load_move_dict()    

def structure_board(board):
    if len(board) != 9:
        raise Exception(f'Board must have length 9, but has length {len(board)}: {board}')
    row1 = [board[0], board[1], board[2]]
    row2 = [board[3], board[4], board[5]]
    row3 = [board[6], board[7], board[8]]
    return [row1, row2, row3]
      
            
    
def play_rb_move(board):
    occupied = {i for i, x in enumerate(board) if x > 0}
    unoccupied = {i for i, x in enumerate(board) if x == 0}
    player = ((board.count(0) + 1) % 2) + 1
    opponent = 2 if player == 1 else 1
    if board.count(0) == 9:
        return choice([0, 2, 4, 6, 8])
    elif board.count(0) == 8:
        if board[4] == 1:
            return choice([0, 2, 6, 8])
        else:
            return 4
    elif board.count(0) == 7:
        return choice(list({0, 2, 4, 6, 8} - occupied))
    else:
        winning_moves = []
        for square in unoccupied:
            board[square] = player
            if check_win_conditions(structure_board(board)) == player:
                winning_moves.append(square)
            board[square] = 0
        if len(winning_moves) > 0:
            return choice(winning_moves)
        blocking_moves = []
        for square in unoccupied:
            board[square] = opponent
            if check_win_conditions(structure_board(board)) == opponent:
                blocking_moves.append(square)
            board[square] = 0
        if len(blocking_moves) > 0:
            return choice(blocking_moves)
        return choice(best_move_dict[tuple(board)])
   
   

def play_ai_move(board):
    moves = []
    for row in range(3):
        for col in range(3):
            moves.append(convert_board_state_to_vector(structure_board(board), (row, col)))
    state = torch.stack(moves)
    unfilled = [int(element==0) for element in board]
    probs = torch.sigmoid(model(state)) * torch.tensor(unfilled)
    return torch.argmax(probs).item()


def play_game(ai_player):
    boards = []
    board = [0,0,0,0,0,0,0,0,0]
    boards.append([x for x in board])
    player = 1
    while check_win_conditions(structure_board(board)) == 0 and not board_full(structure_board(board)):        
        if ai_player == player:
            board[play_ai_move(board)] = player
        else:
            board[play_rb_move(board)] = player
        player = 2 if player == 1 else 1
        boards.append([x for x in board])
    winner = check_win_conditions(structure_board(board))
    if winner != ai_player and winner > 0:
        print("\nai lost!")
        for board in boards:
            print_board(structure_board(board))
            print('\n')
    return winner


ai_wins = 0
rb_wins = 0
ties = 0
for i in range(100):
    ai_player = choice([1, 2])
    result = play_game(ai_player)
    if result == 0:
        ties += 1
    elif result == ai_player:
        ai_wins += 1
    else:
        rb_wins += 1
print(f'{ai_wins}-{rb_wins}-{ties}')
