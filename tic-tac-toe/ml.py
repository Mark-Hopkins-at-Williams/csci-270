import json
from random import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


input_dim = 36
hidden_dim = 32
learning_rate = 1e-3
num_epochs = 20
batch_size = 64

#torch.manual_seed(0)

def convert_board_state_to_vector(board, next_move):
    state = []
    for row in range(3):
        for col in range(3):
            extension = [0., 0., 0., 0.]
            extension[board[row][col]] = 1.
            if next_move == (row, col):
                extension[3] = 1.
            state.extend(extension)
    return torch.tensor(state)

def load_training_data():
    with open('tictactoemoves.json') as reader:
        json_data = json.load(reader)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for datum in json_data:
        add_to_train = random() < 0.9
        X = X_train if add_to_train else X_test
        y = y_train if add_to_train else y_test
        board = datum['board']
        best_moves = datum['best_moves']
        for row in range(3):
            for col in range(3):
                if board[row][col] == 0:
                    next_X = convert_board_state_to_vector(board, (row, col))
                    X.append(next_X)
                    next_y = 1. if [row, col] in best_moves else 0.
                    y.append(torch.tensor(next_y))
        
    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)
    return X_train, y_train, X_test, y_test


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)        
        self.fc4 = nn.Linear(hidden_dim, 1)  # single logit

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  
        x = torch.relu(self.fc3(x))   
        x = self.fc4(x)  # logits
        return x.squeeze(1)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_training_data()
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TwoLayerNet(input_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_steps = 0
    best_model = None
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        with torch.no_grad():
            logits = model(X_test)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            accuracy = (preds == y_test).sum().item() / len(y_test)
            if accuracy >= best_accuracy:
                best_model = model.state_dict()
                best_accuracy = accuracy
            print(f'Test accuracy: {accuracy}')
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            training_steps += 1


    torch.save(best_model, "model.pt")

    model = TwoLayerNet(input_dim, hidden_dim)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        accuracy = (preds == y_test).sum().item() / len(y_test)
        print(f'Test accuracy: {accuracy}')