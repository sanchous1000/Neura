
import numpy as np
from tqdm import tqdm
rng = np.random.default_rng(51)

#
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)
#
def softmax(x):
    e= np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def categorical_cross_entropy(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


class MLP:
    def __init__(self, architecture):
        self.architecture = architecture
        self.depth = len(architecture) - 1
        #
        self.hidden_activation_fn = sigmoid
        self.output_activation_fn = softmax

        # Функция потерь
        self.loss_fn = categorical_cross_entropy
        #
        self.W = self._init_weights(architecture)
        self.b = self._init_biases(architecture)
        self.z = [None] * (self.depth + 1)
        self.a = [None] * (self.depth + 1)
        #
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
    #
    
    def _init_weights(self, arch):
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        return [rng.uniform(-limit, limit + 1e-5, size=(arch[i+1], arch[i])) for i in range(self.depth)]


    #
    def _init_biases(self, arch):
        return [np.zeros((arch[i+1], 1)) for i in range(self.depth)]
    #
    def _feedforward(self, X):
        self.a[0] = X
        
        for i in range(self.depth):
            self.z[i+1] = np.dot(self.a[i], self.W[i].T) + self.b[i].T  
            if i < self.depth - 1:
                self.a[i+1] = self.hidden_activation_fn(self.z[i+1])
            else:
                self.a[i+1] = self.output_activation_fn(self.z[i+1])
    #
    def _backprop(self, y):
        m = y.shape[0]  
        self.delta = self.a[-1] - y
        for i in reversed(range(self.depth)):
            self.dW[i] = np.dot(self.delta.T, self.a[i]) / m
            self.db[i] = np.sum(self.delta, axis=0, keepdims=True).T / m
            if i != 0:
                print(self.delta.shape, self.W[i].shape)
                self.delta = np.dot(self.delta, self.W[i]) * sigmoid_derivative(self.z[i])
    #
    def _compute_loss(self, X, y):
        y_pred = self.predict(X)
        return self.loss_fn(y_pred, y)
    #
    def _update_params(self, lr):
        for i in range(self.depth):
            self.W[i] -= lr * self.dW[i]
            self.b[i] -= lr * self.db[i]
    #
    def train(self, X, y, epochs=1, batch_size=8, lr=1e-2):
        X = np.array(X) 
        y = np.array(y)
        epoch_losses = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            X_batch = X
            y_batch = y
            self._feedforward(X_batch)
            self._backprop(y_batch)
            self._update_params(lr)
            compute_loss = self._compute_loss(X, y)
            epoch_losses.append(compute_loss)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Сompute_loss: {compute_loss}")

        return epoch_losses
    
    #
    def predict(self, X):
        a = X  
        for i in range(self.depth):
            a = np.dot(a, self.W[i].T) + self.b[i].T
            if i < self.depth - 1:
                a = self.hidden_activation_fn(a)
            else:
                a = self.output_activation_fn(a)
        return a  
    

