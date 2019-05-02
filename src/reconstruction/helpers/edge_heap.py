import heapq as hpq
import typing as t
import numpy as np


class Heap():
    def __init__(self, num_elements: int):
        self._heap = []
        self._elements_sum = 0
        self._elements_limit = num_elements

    @property
    def elements_sum(self):
        return self._elements_sum

    def add(self, item: t.Tuple, value: float):
        self._elements_sum += value
        hpq.heappush(self._heap, (value, item))

        if len(self._heap) > self._elements_limit:
            hpq.heappop(self._heap)

    def add_many(self, items: np.array, values: np.array):
        for i, item in enumerate(items):
            w = values[i]
            self.add(item, w)

    def get_sorted(self) -> np.array:
        values = sorted(self._heap, key=lambda x: x[0], reverse=True)
        return np.array(values)
