import typing as t
import numpy as np


# Algorithm: Alias Sampling Method
# helpful tutorial at http://www.keithschwarz.com/darts-dice-coins/

class AliasSampling:
    def __init__(self, probabilities: np.array, classes: np.array) -> None:
        probabilities = self._normalize(probabilities)
        self._precompute(probabilities, classes)

    # region === PUBLIC ===
    def sample(self, size: int = 1) -> t.Any:
        x = np.random.randint(0, len(self._prob_table), size=size)
        y = np.random.random_sample(size)

        replace_idx = y > self._prob_table[x]
        x[replace_idx] = self._alias_table[x][replace_idx]
        return self._classes[x]

    def random_sample(self):
        l = len(self._classes)
        r = np.random.randint(l)
        return self._classes[r]

    # endregion

    # region === PRIVATE ===
    def _normalize(self, probabilities: np.array) -> np.array:
        sum_p = np.sum(probabilities)
        if sum_p > .99:
            return probabilities
        else:
            return probabilities / sum(probabilities)

    def _precompute(self, probabilities: np.array, classes: np.array) -> None:
        prob_table = np.zeros_like(probabilities)
        alias_table = np.zeros_like(probabilities, dtype=int)

        # Multiply probabilities so that mean_p has value 1
        probabilities = probabilities * len(probabilities)

        # Divide probabilities into groups
        less_than_one = []
        more_than_one = []
        for i, pr in enumerate(probabilities):
            if pr < 1:
                less_than_one.append((i, pr))

            else:
                more_than_one.append((i, pr))

        # Select one from each group
        while more_than_one and less_than_one:
            l_i, l_pr = less_than_one.pop()
            m_i, m_pr = more_than_one.pop()

            # Complete another column (lower part is from less than 1, upper from more than 1)
            prob_table[l_i] = l_pr
            alias_table[l_i] = m_i

            # Lower the [more_than_one]'s probability by piece added to alias
            m_pr = (m_pr + l_pr) - 1
            if m_pr < 1:
                less_than_one.append((m_i, m_pr))
            else:
                more_than_one.append((m_i, m_pr))

        for i, pr in more_than_one:
            prob_table[i] = pr
            alias_table[i] = i

        for i, pr in less_than_one:
            prob_table[i] = pr
            alias_table[i] = i

        # Assign private variables
        self._prob_table = np.array(prob_table)
        self._alias_table = np.array(alias_table)
        self._classes = classes
    # endregion
