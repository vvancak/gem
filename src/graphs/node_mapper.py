class NodeMapper():
    def __init__(self):
        self._reversed = []
        self._mapping = {}
        self._it = 0

    def __getitem__(self, key: str):
        # Try get mapping
        value = self._mapping.get(key, self._it)

        # Create if not exists
        if (value == self._it):
            self._mapping[key] = self._it
            self._reversed.append(key)
            self._it += 1

        return value

    @property
    def reversed_mapping(self):
        return self._reversed

    @property
    def mapping(self):
        return self._mapping

    def node_ids_sorted(self):
        return sorted(self._mapping, key=self._mapping.get)
