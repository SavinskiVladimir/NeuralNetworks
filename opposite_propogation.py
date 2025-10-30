import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KohonenCard:
    def __init__(self):
        self.data = None
        self.W = np.array(list([np.random.random(), np.random.random(), np.random.random()] for _ in range(20)))
        grid = [[0, 0, 0, 0] for _ in range(5)]
        for i in range(5):
            for j in range(4):
                grid[i][j] = self.W[i + j]
        self.W_grid = grid
        self.eta = 0.8
        self.sigma = 1
        self.r = 3

    def generate_data(self, num_points):
        self.data = []
        for _ in range(num_points):
            R, G, B = np.random.randint(256), np.random.randint(256), np.random.randint(256)
            self.data.append((R, G, B))

        self.data = np.array(self.data)

    def normalize_data(self):
        self.data = self.data / 255.0

    def decrease_eta(self):
        self.eta -= 0.08

    def decrease_sigma(self):
        self.sigma += 0.08

    def g(self, r1, r2):
        return np.linalg.norm([r1, r2]) / (2 * self.sigma ** 2)

    def train(self):
        self.normalize_data()
        epoch_count = 1
        while self.eta > 0:
            print(f"\nЭпоха №{epoch_count}, eta: {self.eta}")
            for X in self.data:
                Y = np.dot(self.W, X)
                j = np.argmax(Y)
                for k1 in range(5):
                    for k2 in range(4):
                        self.W_grid[k1][k2] += self.g(self.W_grid[k1][k2], self.W[j]) * self.eta * (X - self.W_grid[k1][k2])
                        self.W[k1 + k2] += self.g(self.W[k1 + k2], self.W[j]) * self.eta * (X - self.W[k1 + k2])
                print(f"Номер нейрона-победителя: {j}")

            epoch_count += 1
            self.decrease_eta()
            self.decrease_sigma()

    def visualize_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = self.data[:, 0]
        ys = self.data[:, 1]
        zs = self.data[:, 2]
        colors = self.data

        ax.scatter(xs, ys, zs, c=colors, marker='o')
        ax.set_xlabel('Красный')
        ax.set_ylabel('Зелёный')
        ax.set_zlabel('Синий')
        ax.set_title('Обучающие данные')

    def visualize_competitive_neurons(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        neuron_colors = self.W
        ax.scatter(self.W[:, 0], self.W[:, 1], self.W[:, 2], c=neuron_colors, marker='o', s=100)

        dists = np.linalg.norm(self.data[:, None, :] - self.W[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        data_colors = neuron_colors[labels]

        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=data_colors, marker='.', alpha=0.5)
        ax.set_xlabel('Красный')
        ax.set_ylabel('Зелёный')
        ax.set_zlabel('Синий')
        ax.set_title('Результат обучения')
        plt.show()

class Grossberg:
    def __init__(self, input_dim, output_dim):
        self.V = np.random.rand(output_dim, input_dim)
        self.beta = 0.05

    def train(self, kohonen, epochs=10):
        for epoch in range(epochs):
            for X in kohonen.data:
                dists = np.linalg.norm(kohonen.W - X, axis=1)
                winner = np.argmin(dists)
                self.V[winner] += self.beta * (X - self.V[winner])
            self.beta *= 0.8

    def visualize_competitive_neurons(self, KohonnedCard):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        neuron_colors = KohonnedCard.W * self.V
        ax.scatter(KohonnedCard.W[:, 0] * self.V[:, 0], KohonnedCard.W[:, 1] * self.V[:, 1], KohonnedCard.W[:, 2] * self.V[:, 2], c=neuron_colors, marker='o', s=100)

        dists = np.linalg.norm(KohonnedCard.data[:, None, :] - KohonnedCard.W[None, :, :] * self.V[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        data_colors = neuron_colors[labels]

        ax.scatter(KohonnedCard.data[:, 0], KohonnedCard.data[:, 1], KohonnedCard.data[:, 2], c=data_colors, marker='.', alpha=0.5)
        ax.set_xlabel('Красный')
        ax.set_ylabel('Зелёный')
        ax.set_zlabel('Синий')
        ax.set_title('Результат обучения')
        plt.show()

K = KohonenCard()
K.generate_data(200)
K.train()

G = Grossberg(input_dim=3, output_dim=20)
G.train(K, epochs=10)

K.visualize_data()
K.visualize_competitive_neurons()
G.visualize_competitive_neurons(K)
plt.show()

