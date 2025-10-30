import numpy as np
import matplotlib.pyplot as plt
from random import choices, random


class ElmanRNN:
    def __init__(self, seq_len=20, num_neurons=40):
        self.seq_len = seq_len
        self.data_step = 0.02
        self.data_limit = 10
        self.generate_data()

        limit_U = np.sqrt(6 / (1 + num_neurons))
        limit_W = np.sqrt(6 / (num_neurons + num_neurons))
        limit_V = np.sqrt(6 / (num_neurons + 1))

        self.U = np.random.uniform(-limit_U, limit_U, (num_neurons, 1))
        self.W = np.random.uniform(-limit_W, limit_W, (num_neurons, num_neurons))
        self.V = np.random.uniform(-limit_V, limit_V, (1, num_neurons))
        self.bh = np.zeros((num_neurons, 1))
        self.by = np.zeros((1, 1))

        self.h = np.zeros((num_neurons, 1))

    def func(self, x):
        return 10 * np.exp(-0.2 * x) * np.cos(10 * x)

    def generate_data(self):
        x = np.arange(0, self.data_limit, self.data_step)
        y = self.func(x)

        self.orig_min = y.min()
        self.orig_max = y.max()

        y = (y - self.orig_min) / (self.orig_max - self.orig_min) - 0.5
        noise_indecies = choices(range(len(x)), k=(len(x) // 8))
        for i in noise_indecies:
            y[i] += random() * y[i] - y[i] / 2

        self.x_vals = x
        self.data = y.reshape(-1, 1)

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x_seq):
        T = len(x_seq)
        h_states = []
        h = self.h.copy()
        outputs = []

        for t in range(T):
            x_t = x_seq[t].reshape(1, 1)
            h = self.tanh(self.U @ x_t + self.W @ h + self.bh)
            y_hat = self.V @ h + self.by
            outputs.append(y_hat)
            h_states.append(h.copy())

        self.h = h.copy()
        return outputs, h_states

    def backprop(self, x_seq, y_true_seq, lr=0.01):
        outputs, h_states = self.forward(x_seq)
        T = len(x_seq)

        errors = [outputs[t] - y_true_seq[t] for t in range(T)]
        loss = np.mean([0.5 * e**2 for e in errors])

        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(self.h)

        for t in reversed(range(T)):
            dy = errors[t]
            dV += dy @ h_states[t].T
            dby += dy

            dh = self.V.T @ dy + dh_next
            da = dh * (1 - h_states[t] ** 2)
            dbh += da
            x_t = x_seq[t].reshape(1, 1)
            dU += da @ x_t.T
            h_prev = h_states[t - 1] if t > 0 else np.zeros_like(self.h)
            dW += da @ h_prev.T
            dh_next = self.W.T @ da

        self.U -= lr * dU
        self.W -= lr * dW
        self.V -= lr * dV
        self.bh -= lr * dbh
        self.by -= lr * dby

        return loss.item()

    def train(self, epochs=300, lr=0.01):
        losses = []
        for e in range(epochs):
            loss = 0
            for t in range(len(self.data) - self.seq_len - 1):
                x_seq = self.data[t:t+self.seq_len]
                y_seq = self.data[t+1:t+self.seq_len+1]
                self.h = np.zeros_like(self.h)
                loss += self.backprop(x_seq, y_seq, lr)
            loss /= (len(self.data) - self.seq_len)
            losses.append(loss)
            if (e + 1) % 5 == 0:
                print(f"Epoch {e+1}/{epochs} — loss: {loss:.6f}")

    def visualize(self):
        start_x = self.data_limit / 2
        x_pred = np.arange(start_x, 12, self.data_step)

        seq_inputs = np.array([self.func(x) for x in x_pred])
        seq_inputs = (seq_inputs - self.orig_min) / (self.orig_max - self.orig_min) - 0.5
        seq_inputs = seq_inputs.reshape(-1, 1)

        preds = []

        for i in range(len(seq_inputs)):
            y_hat, _ = self.forward(seq_inputs[i:i + 1])
            preds.append(y_hat[-1].item())

        preds_denorm = (np.array(preds) + 0.5) * (self.orig_max - self.orig_min) + self.orig_min
        data_denorm_x = self.x_vals
        data_denorm_y = (self.data.flatten() + 0.5) * (self.orig_max - self.orig_min) + self.orig_min

        plt.figure(figsize=(10, 5))
        plt.plot(data_denorm_x, data_denorm_y, label='Обучающие данные')
        plt.plot(x_pred, preds_denorm, 'r', label='RNN прогноз')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()

rnn = ElmanRNN(seq_len=50, num_neurons=30)
rnn.train(epochs=40, lr=0.005)
rnn.visualize()
