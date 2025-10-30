import numpy as np

class Node:
    def __init__(self, n_connections):
        limit = np.sqrt(6 / (n_connections + 1))
        self.w = np.random.uniform(-limit, limit, n_connections)
        self.t = np.random.uniform(-limit, limit)

    def sigma(self, x):
        return 1 / (1 + np.exp(-(sum(self.w * x) + self.t)))

class Perceptron:
    def __init__(self):
        # два скрытых слоя по 3 и 4 нейрона и 1 выходной нейрон
        self.layers = [[Node(3), Node(3), Node(3)], [Node(3), Node(3), Node(3), Node(3)], [Node(4)]]
        # случайно заданная функция от 3 переменных
        self.examples = [[0, 0, 0, 1],
                         [0, 0, 1, 1],
                         [0, 1, 0, 0],
                         [0, 1, 1, 1],
                         [1, 0, 0, 0],
                         [1, 0, 1, 0],
                         [1, 1, 0, 0],
                         [1, 1, 1, 1]]
        self.lr = 0.1

    def get_error(self, y, d):
        return 0.5 * (y - d) ** 2

    def count_F(self, example):
        data, result = example[:-1], example[-1]
        outs = []  # для хранения значения сигмы на всех нейронах
        for layer in self.layers:
            out = []  # для хранения значения сигмы на текущем слое
            for node in layer:
                out.append(node.sigma(data))
            data = out
            outs.append(out)

        return [outs, data]

    def study(self):
        flag = False
        for epoch in range(200000):
            total_error = 0
            for example in self.examples:
                # расчёт функции
                data, result = example[:-1], example[-1]
                F = self.count_F(example)
                outs, data = F[0], F[1]

                # расчёт ошибки
                e = self.get_error(data[0], result)
                total_error += e

                # расчёт дельт
                deltas = [[] for _ in range(len(self.layers))]
                deltas[-1].append((result - data[0]) * data[0] * (1 - data[0])) # дельта для выходного слоя
                for i in range(len(self.layers) - 2, -1, -1):
                    for j, node in enumerate(self.layers[i]):
                        out_h = outs[i][j]
                        connected_weights = [self.layers[i + 1][k].w[j] for k in range(len(self.layers[i + 1]))]
                        delta_next = deltas[i + 1]
                        s = sum(w * d for w, d in zip(connected_weights, delta_next))
                        deltas[i].append(out_h * (1 - out_h) * s)

                # изменение весов
                for i in range(len(self.layers)):
                    for j in range(len(self.layers[i])):
                        if i == 0:
                            inputs = example[:-1]
                        else:
                            inputs = outs[i - 1]
                        for k in range(len(self.layers[i][j].w)):
                            self.layers[i][j].w[k] += self.lr * deltas[i][j] * inputs[k]
                        self.layers[i][j].t += self.lr * deltas[i][j]

            if epoch % 1000 == 0:
                print(f"Эпоха: {epoch}, Ошибка: {total_error}")

                # проверка условия остановки
            if total_error <= 0.005:
                print(f"Обучение завершено на эпохе {epoch}, ошибка: {total_error}\n")
                flag = True
                break

            if flag:
                break

p = Perceptron()
p.study()
print(f"x1: 0, x2: 0, x3 = 0, y = 1, ответ: {p.count_F([0, 0, 0, 1])[1]}")
print(f"x1: 0, x2: 0, x3 = 1, y = 1, ответ: {p.count_F([0, 0, 1, 1])[1]}")
print(f"x1: 0, x2: 1, x3 = 0, y = 0, ответ: {p.count_F([0, 1, 0, 0])[1]}")
print(f"x1: 0, x2: 1, x3 = 1, y = 1, ответ: {p.count_F([0, 1, 1, 1])[1]}")
print(f"x1: 1, x2: 0, x3 = 0, y = 0, ответ: {p.count_F([1, 0, 0, 0])[1]}")
print(f"x1: 1, x2: 0, x3 = 1, y = 0, ответ: {p.count_F([1, 0, 1, 0])[1]}")
print(f"x1: 1, x2: 1, x3 = 0, y = 0, ответ: {p.count_F([1, 1, 0, 0])[1]}")
print(f"x1: 1, x2: 1, x3 = 1, y = 1, ответ: {p.count_F([1, 1, 1, 1])[1]}")












