class NodeMapper():
    def __init__(self):
        self.mapping = {}
        self._it = 0

    def __getitem__(self, key: str):
        # Try get mapping
        value = self.mapping.get(key, self._it)

        # Create if not exists
        if (value == self._it):
            self.mapping[key] = self._it
            self._it += 1

        return value

    def node_ids_sorted(self):
        return sorted(self.mapping, key=self.mapping.get)
