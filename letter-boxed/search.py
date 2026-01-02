from abc import ABC, abstractmethod
from queue import LifoQueue, Queue
from queue import PriorityQueue
import random
from tqdm import tqdm

random.seed(48)


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


def search_template(
    space, container, heuristic_fn=lambda state, space: 0, memoize=False
):
    """General-purpose algorithmic template for search, e.g. DFS or BFS.

    Parameters
    ----------
    space : SearchSpace
        The search space
    container : queue.Queue or queue.LifoQueue or PriorityQueueWithFunction (defined below)
        The container for processing nodes of the search tree.
    heuristic_fn : function that takes a search state and a SearchSpace and returns a non-negative number
        The heuristic function (defaults to a function that always returns zero)
    """
    visited = set()
    initial_node = (space.get_start_state(), 0, 0, tuple())
    container.put(initial_node)
    count = 0
    progress_bar = tqdm()
    while not container.empty():
        (q, g, h, solution) = container.get()
        if q not in visited:
            count += 1
            progress_bar.update(1)
            if memoize:
                visited.add(q)
            if space.is_final_state(q):
                progress_bar.close()
                print(f"Search nodes visited: {count}")
                return solution
            successors = space.get_successors(q)
            for next_state, action, cost in successors:
                h = heuristic_fn(next_state, space)
                successor_node = (next_state, g + cost, h, solution + (action,))
                container.put(successor_node)


def depth_first_search(problem, memoize=False):
    return search_template(problem, LifoQueue(), memoize=memoize)


def breadth_first_search(problem, memoize=False):
    return search_template(problem, Queue(), memoize=memoize)


class PriorityQueueWithFunction:
    def __init__(self, priority_fn):
        self.queue = PriorityQueue()
        self.priority_fn = priority_fn

    def put(self, item):
        prioritized_item = (self.priority_fn(item), item)
        self.queue.put((prioritized_item, item))

    def get(self):
        _, item = self.queue.get()
        return item

    def empty(self):
        return self.queue.empty()


def uniform_cost_search(problem, memoize=False):
    return search_template(
        problem, PriorityQueueWithFunction(lambda x: x[1]), memoize=memoize
    )


def a_star_search(problem, heuristic, memoize=False):
    return search_template(
        problem,
        PriorityQueueWithFunction(lambda x: x[1] + x[2]),
        heuristic,
        memoize=memoize,
    )
