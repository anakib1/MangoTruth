import math


class DnaSerializableFunction:
    def calc(self, x: int) -> float:
        pass


class SublogarithmFunction(DnaSerializableFunction):
    def calc(self, x: int) -> float:
        return x * math.log(x)
