import json
from random import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from connectfour import print_board

input_dim = 126  # 42  # 126
hidden_dim = 32
learning_rate = 1e-3
num_epochs = 200
batch_size = 64
num_outcomes = 7

# torch.manual_seed(0)


def convert_board_state_to_vector(board):
    num_rows = len(board)
    num_cols = len(board[0])
    # encodings = {0: 0.0, 1: 1.0, 2: -1.0}
    state = []
    for row in range(num_rows):
        for col in range(num_cols):
            # state.append(encodings[board[row][col]])

            extension = [0.0, 0.0, 0.0]
            extension[board[row][col]] = 1.0
            state.extend(extension)
    return torch.tensor(state)


def load_training_data():
    with open("c4moves.6ply.json") as reader:
        json_data = json.load(reader)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for datum in json_data:
        add_to_train = random() < 0.9
        X = X_train if add_to_train else X_test
        y = y_train if add_to_train else y_test
        board = datum["board"]
        move = datum["move"]
        vec = convert_board_state_to_vector(board)
        X.append(vec)
        y.append(torch.tensor(move))

    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)
    return X_train, y_train, X_test, y_test


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_outcomes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_outcomes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_training_data()
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TwoLayerNet(input_dim, hidden_dim, num_outcomes)
    criterion = nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_steps = 0
    best_model = None
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        with torch.no_grad():
            logits = model(X_test)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_test).sum().item() / len(y_test)
            if accuracy >= best_accuracy:
                best_model = model.state_dict().copy()
                best_accuracy = accuracy
            print(f"Test accuracy: {accuracy}")
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            training_steps += 1

    torch.save(best_model, "model.pt")

    model = TwoLayerNet(input_dim, hidden_dim, num_outcomes)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y_test).sum().item() / len(y_test)
        print(f"Final test accuracy: {accuracy}")
