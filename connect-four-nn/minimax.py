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


def create_ranked_minimax_function(search_space, evaluation_fn, player, max_plys):
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

    def root_minimax(state):
        evaluated_actions = []
        for action, next_state in search_space.get_successors(state):
            _, child_value = minimax(next_state, plys_left=max_plys - 1)
            evaluated_actions.append((child_value, action))
        return sorted(evaluated_actions, key=lambda x: -x[0])

    return root_minimax
