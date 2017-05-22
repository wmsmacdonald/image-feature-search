

class Partitioned:
    def __init__(self):
        self.partitions = []
        self.values = []

    # partition is a range
    # value is any object
    def add_partition(self, length, value):
        if len(self.partitions) == 0:
            self.partitions.append(range(0, length))
        else:
            base_position = self.partitions[-1][-1] + 1
            self.partitions.append(
                range(base_position, base_position + length)
            )
        self.values.append(value)

    def get_value(self, position):
        for r, v in zip(self.partitions, self.values):
            if position in r:
                return v
        else:
            raise IndexError
