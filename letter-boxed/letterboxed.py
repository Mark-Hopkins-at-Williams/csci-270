from search import SearchSpace, uniform_cost_search
from search import a_star_search, depth_first_search, breadth_first_search


class Trie:

    def __init__(self, words, prefix=""):
        self.is_accepting_node = False
        suffixes = dict()
        for word in words:
            if len(word) > 0:
                if word[0] not in suffixes:
                    suffixes[word[0]] = []
                suffixes[word[0]].append(word[1:])
            else:
                self.is_accepting_node = True
        self.children = {
            letter: Trie(suffixes[letter], prefix + letter) for letter in suffixes
        }
        self.prefix = prefix

    def __contains__(self, word):
        if len(word) == 0:
            return self.is_accepting_node
        elif word[0] in self.children:
            return self.children[word[0]].__contains__(word[1:])
        else:
            return False

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __repr__(self):
        return f"Trie({self.prefix})"


class LetterBoxedSearchSpace(SearchSpace):

    def __init__(self, letters, words):
        self.letters = letters
        self.word_trie = Trie(words)

    def get_start_state(self):
        return (self.word_trie, None, tuple([0] * len(self.letters)))

    def is_final_state(self, state):
        trie_node, _, used_letters = state
        return trie_node in self.word_trie.children.values() and 0 not in used_letters

    def get_successors(self, state):
        trie_node, prev_letter_index, used_letters = state
        successors = []
        for i, letter in enumerate(self.letters):
            if prev_letter_index is None or prev_letter_index // 3 != i // 3:
                if letter in trie_node.children:
                    new_used_letters = tuple(
                        [b if j != i else 1 for j, b in enumerate(used_letters)]
                    )
                    successor_state = (trie_node.children[letter], i, new_used_letters)
                    successor = (successor_state, letter, 0)
                    successors.append(successor)
        if (
            trie_node.is_accepting_node
            and self.letters[prev_letter_index] in self.word_trie.children
        ):
            successor_state = (
                self.word_trie.children[self.letters[prev_letter_index]],
                prev_letter_index,
                used_letters,
            )
            successor = (successor_state, "ENTER", 1)
            successors.append(successor)
        return successors


def is_spellable(word, letters):
    if len(word) == 0 or word[0] not in letters:
        return False
    else:
        for i in range(1, len(word)):
            if word[i] not in letters:
                return False
            else:
                prev_index = letters.index(word[i - 1])
                this_index = letters.index(word[i])
                if prev_index // 3 == this_index // 3:
                    return False
    return True


def create_heuristic(letters, words):
    from math import ceil

    def my_heuristic(state, space):
        _, _, used_letters = state
        return ceil(used_letters.count(0) / max_new_letters)

    letter_set = set(letters)
    max_new_letters = 0
    for word in words:
        if (
            len(set(word) - letter_set) == 0
        ):  # i.e., the word can be spelled with the given letters
            new_letters = set(word) & letter_set
            if len(new_letters) > max_new_letters:
                max_new_letters = len(new_letters)

    print(f"Created heuristic with max new letters: {max_new_letters}")
    return my_heuristic


def display_solution(solution):
    assert solution[-1] == "ENTER"
    result = []
    for action in solution[:-1]:
        if action == "ENTER":
            last_letter = result[-1]
            result.append(" ")
            result.append(last_letter)
        else:
            result.append(action)
    print("".join(result))


if __name__ == "__main__":
    from itertools import combinations

    # letters = list("luarpeoxmgiq")
    # letters = list("abcdefghijkm")
    letters = list("raghltieocyp")
    words = []
    with open("words.scrabble.txt") as reader:
        for line in reader:
            word = line.strip()
            if is_spellable(word, letters):
                words.append(word)

    space = LetterBoxedSearchSpace(letters, words)
    solution = a_star_search(space, create_heuristic(letters, words), memoize=True)
    # solution = depth_first_search(space, memoize=True)
    # solution = breadth_first_search(space, memoize=True)
    display_solution(solution)
