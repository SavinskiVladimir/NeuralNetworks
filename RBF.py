import numpy as np
import matplotlib.pyplot as plt


class RBFnet:
    def __init__(self, num_centers):
        self.train = None
        self.valid = None
        self.centers = None
        self.sigma = None
        self.W = None
        self.num_centers = num_centers
        self.num_classes = 3

    def generate_train(self, train_size):
        x = np.random.rand(train_size)
        y = np.random.rand(train_size)
        train_data = []
        for i in range(train_size):
            f = 0
            if y[i] >= x[i] and y[i] >= -0.5 * x[i] + 0.5:
                f = 1
            elif y[i] < x[i] and y[i] < -0.5 * x[i] + 0.5:
                f = 2
            train_data.append([x[i], y[i], f])

        noise_size = train_size // 10
        for i in range(noise_size):
            train_data.append([np.random.rand(), np.random.rand(), np.random.randint(0, 3)])

        self.train = np.array(train_data)
        np.random.shuffle(self.train)

    def generate_valid(self, valid_size):
        x = np.random.rand(valid_size)
        y = np.random.rand(valid_size)
        valid_data = []
        for i in range(valid_size):
            f = 0
            if y[i] >= x[i] and y[i] >= -0.5 * x[i] + 0.5:
                f = 1
            elif y[i] < x[i] and y[i] < -0.5 * x[i] + 0.5:
                f = 2
            valid_data.append([x[i], y[i], f])
        self.valid = np.array(valid_data)

    def kmeans(self, data, k, max_iters=100):
        centers = data[np.random.choice(len(data), k, replace=False)]
        for _ in range(max_iters):
            distances = np.linalg.norm(data[:, None] - centers[None, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([
                data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                for i in range(k)
            ])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        return centers

    def calc_sigma(self):
        sigmas = np.zeros(self.num_centers)
        for i in range(self.num_centers):
            dists = np.linalg.norm(self.centers - self.centers[i], axis=1)
            dists = np.sort(dists[dists > 0])
            if len(dists) > 0:
                sigmas[i] = np.mean(dists[:min(3, len(dists))])
            else:
                sigmas[i] = 1.0
        return sigmas

    def rbf(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for ci, (c, s) in enumerate(zip(self.centers, self.sigma)):
            diff = X - c
            G[:, ci] = np.exp(-np.sum(diff ** 2, axis=1) / (2 * s ** 2))
        return G

    def train_rbf(self):
        X = self.train[:, :2]
        y = self.train[:, 2].astype(int)
        self.centers = self.kmeans(X, self.num_centers)
        self.sigma = self.calc_sigma()
        G = self.rbf(X)
        Y = np.zeros((len(y), self.num_classes))
        Y[np.arange(len(y)), y] = 1
        self.W, _, _, _ = np.linalg.lstsq(G, Y, rcond=None)

    def predict(self):
        X = self.valid[:, :2]
        G = self.rbf(X)
        outputs = G.dot(self.W)
        preds = np.argmax(outputs, axis=1)
        return preds

    def compare_predictions(self):
        true_labels = self.valid[:, 2].astype(int)
        pred_labels = self.predict()
        for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
            print(f"Точка {i}: истинный класс = {true}, предсказанный класс = {pred}")
        accuracy = np.mean(true_labels == pred_labels)
        print(f"Точность на валидационных данных: {accuracy:.4f}")

    def visualize(self):
        x, y, f = self.train[:, 0], self.train[:, 1], self.train[:, 2]
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=f, alpha=0.7)
        plt.colorbar(scatter, ticks=[0, 1, 2], label='Класс')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Обучающие данные')

    def visualize_valid(self):
        x, y, true_labels = self.valid[:, 0], self.valid[:, 1], self.valid[:, 2].astype(int)
        pred_labels = self.predict()
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=true_labels, marker='o', edgecolors='k', cmap='viridis', alpha=0.5)
        plt.scatter(x, y, c=pred_labels, marker='x', cmap='viridis')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Сравнение истинных и предсказанных классов на валидации')
        plt.colorbar(ticks=[0, 1, 2])

rbf = RBFnet(num_centers=15)
rbf.generate_train(300)
rbf.generate_valid(50)
rbf.train_rbf()
rbf.compare_predictions()
rbf.visualize()
rbf.visualize_valid()
plt.show()