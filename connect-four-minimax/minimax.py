def create_minimax_function(search_space, evaluation_fn, player, max_plys):
    def minimax(state, alpha=float("-inf"), beta=float("inf"), plys_left=max_plys):
        whose_turn, _ = state
        if search_space.is_final_state(state) or plys_left == 0:
            return None, evaluation_fn(state, player)
        best_value = float("-inf") if whose_turn == player else float("inf")
        best_action = None
        for action, next_state in search_space.get_successors(state):
            if alpha >= beta:
                return best_action, best_value
            _, child_value = minimax(next_state, alpha, beta, plys_left=plys_left - 1)
            if whose_turn == player:
                if child_value > best_value:
                    best_value = child_value
                    best_action = action
                alpha = max(alpha, child_value)
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_action = action
                beta = min(beta, child_value)
        return best_action, best_value

    return minimax


if __name__ == "__main__":

    minimax_fn = create_minimax_function(
        ConnectFourSearchSpace(6, 7), connect_four_evaluation_fn, player=1, max_plys=6
    )
    # row1 = [0, 0, 0]
    # row2 = [0, 1, 0]
    # row3 = [0, 2, 0]
    # state = (1, [row1, row2, row3])

    row1 = [0, 0, 0, 0, 0, 0, 0]
    row2 = [0, 0, 0, 0, 0, 0, 0]
    row3 = [0, 0, 0, 0, 0, 0, 2]
    row4 = [1, 1, 1, 2, 2, 2, 1]
    row5 = [2, 2, 2, 1, 2, 1, 2]
    row6 = [1, 2, 1, 1, 2, 1, 1]
    state = (1, [row1, row2, row3, row4, row5, row6])

    print(minimax_fn(state))
