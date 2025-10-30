import numpy as np


class NeuralNode:
    def __init__(self):
        self.w = np.zeros(2)
        self.examples = np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]])
        self.t = 4

    def count_F(self, x):
        return 1 if np.dot(np.array(x), self.w) + (-1) * self.t > 0 else -1

    def change_weight(self, iter_count):
        i = 0
        count_true = 0
        while i < len(self.examples):
            print(f"Итерация {iter_count}, веса: {self.w}")
            F = self.count_F(self.examples[i][:-1])
            if F == self.examples[i][-1]:
                count_true += 1
            else:
                self.w = np.array([self.w[j] + self.examples[i][j] for j in range(len(self.w))])
                self.t = self.t - self.examples[i][-1]
            i += 1
            iter_count += 1

        return count_true, iter_count

    def study(self):
        iter_count = 1
        count_true, iter_count = self.change_weight(iter_count)
        while count_true != len(self.examples):
            count_true, iter_count = self.change_weight(iter_count)




n = NeuralNode()
n.study()
print(f"\nВход: -1, -1; ожидаемое значение: -1; реальное:{n.count_F([-1, -1])}")
print(f"Вход: -1, 1; ожидаемое значение: 1; реальное:{n.count_F([-1, 1])}")
print(f"Вход: 1, -1; ожидаемое значение: 1; реальное:{n.count_F([1, -1])}")
print(f"Вход: 1, 1; ожидаемое значение: 1; реальное:{n.count_F([1, 1])}")