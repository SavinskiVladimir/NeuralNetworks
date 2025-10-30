import numpy as np


class NeuralNode:
    def __init__(self):
        self.w = np.zeros(2)
        self.examples = np.array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
        self.lr = 0.25
        self.t = 0
        self.e = 0

    def count_F(self, x):
        return 1 if np.dot(np.array(x), self.w) + (-1) * self.t > 0 else -1

    def check(self):
        for i in range(len(self.examples)):
            F = self.count_F(self.examples[i][:-1])
            e_model = self.examples[i][-1] - F
            if e_model != self.e:
                return False

        return True

    def study(self):
        for epoch in range(1000):
            for i in range(len(self.examples)):
                F = self.count_F(self.examples[i][:-1])
                e_model = self.examples[i][-1] - F
                print(f"Эпоха: {epoch}, OUT: {F}, y: {self.examples[i][-1]}, w: {self.w}, t: {self.t}, e: {e_model}")
                if e_model != self.e:
                    self.w += np.array(self.examples[i][:-1]) * self.lr * e_model
                    self.t += (-1) * self.lr * e_model

            if self.check():
                break


n = NeuralNode()
n.study()
print(f"\nx1: -1, x2: -1, y = -1, ответ: {n.count_F([-1, -1])}")
print(f"x1: -1, x2: -1, y = 1, ответ: {n.count_F([-1, 1])}")
print(f"x1: 1, x2: -1, y = 1, ответ: {n.count_F([1, -1])}")
print(f"x1: 1, x2: 1, y = 1, ответ: {n.count_F([1, 1])}")
