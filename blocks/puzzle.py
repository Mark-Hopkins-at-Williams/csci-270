from search import SearchSpace, bfs

class BlockPuzzleSearchSpace(SearchSpace):

    def __init__(self, intervals, cube_width):
        super().__init__()
        self.intervals = intervals
        self.cube_width = cube_width
        raise NotImplementedError('Add more to me, as needed!')

    def get_start_state(self):
        """Returns the start state.

        A state of this search space is a sequence of directions. The start state
        contains a single arbitrary initial direction ('E').

        Returns
        -------
        tuple[str]
            The start state
        """
        raise NotImplementedError('Implement me!')

    def is_final_state(self, state):
        """Checks whether a given state is a final state.

        To qualify as a final state, the state trajectory should visit all
        positions in a 3x3 cube (without visiting the same position twice).

        Parameters
        ----------
        state : tuple[str]
            A state of the search space, i.e. a sequence of directions

        Returns
        -------
        bool
            True iff the state is a final state
        """
        raise NotImplementedError('Implement me!')

    def get_successors(self, state):
        """Determines the possible successors of a state.

        A state is a sequence of directions. To generate its successor, we append a direction
        that forces the puzzle to make a 90-degree turn along some axis. In other words,
        one cannot append the direction in which the puzzle is already heading, nor can one
        append the completely opposite direction.

        For instance, if the state is (U, N, W), then we cannot append directions "W" (the
        direction in which the puzzle is currently going) or "E" (the opposite direction)
        to derive a successor.
        
        Parameters
        ----------
        state : tuple[str]
            A state of the search space, i.e. a sequence of directions

        Returns
        -------
        list[tuple[str]]
            The list of valid successor states.
        """
        raise NotImplementedError('Implement me!')
       

def construct_search_space_for_2x2x2_puzzle():
    return BlockPuzzleSearchSpace(intervals=(1, 1, 1, 1, 1, 1, 1), cube_width=2)


def construct_search_space_for_3x3x3_puzzle():
    return BlockPuzzleSearchSpace(intervals=(2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2), cube_width=3)


def small_solution():
    space = construct_search_space_for_2x2x2_puzzle()
    return bfs(space)


def puzzle_solution():
    space = construct_search_space_for_3x3x3_puzzle()
    return bfs(space)
