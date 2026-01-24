from queue import LifoQueue, Queue
from abc import ABC, abstractmethod

class SearchSpace(ABC):

    @abstractmethod
    def get_start_state(self):
        """Returns the start state.

        Returns
        -------
        tuple[str]
            The start state
        """

    @abstractmethod
    def is_final_state(self, state):
        """Checks whether a given state is a final state.

        Parameters
        ----------
        state : tuple[str]
            A state of the search space

        Returns
        -------
        bool
            True iff the state is a final state
        """

    @abstractmethod
    def get_successors(self, state):
        """Determines the possible successors of a state.

        Parameters
        ----------
        state : tuple[str]
            A state of the search space

        Returns
        -------
        list[tuple[str]]
            The list of valid successor states
        """


def search_template(space, container):
    """General-purpose algorithmic template for search, e.g. DFS or BFS.

    Parameters
    ----------
    space : SearchSpace
        The search space
    container : queue.Queue or queue.LifoQueue
        The container for processing nodes of the search tree.
    """

    container.put(space.get_start_state())
    count = 0
    while not container.empty():
        count += 1
        next_state = container.get()
        if space.is_final_state(next_state):
            print(f"Search nodes visited: {count}")
            return next_state
        successors = space.get_successors(next_state)
        for successor in successors:
            container.put(successor)


def bfs(space):
    """Runs breadth-first search (BFS) on a search space.

    Parameters
    ----------
    space : SearchSpace
        The search space
    """
    return search_template(space, Queue())


def dfs(space):
    """Runs depth-first search (DFS) on a search space.

    Parameters
    ----------
    space : SearchSpace
        The search space
    """
    return search_template(space, LifoQueue())

