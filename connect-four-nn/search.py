from abc import ABC, abstractmethod


class SearchSpace(ABC):

    @abstractmethod
    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """

    @abstractmethod
    def is_final_state(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """

    @abstractmethod
    def get_successors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
