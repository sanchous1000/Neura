
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
def binary_cross_entropy(y_pred, y_true):
    y_pred = np.clip(y_pred,  1e-15  , 1 -  1e-15  )
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class MLP:
    def __init__(self, architecture):
        self.architecture = architecture
        self.depth = len(architecture) - 1
        #
        self.activation_fn = sigmoid
        self.activation_dfn = sigmoid_derivative
        self.loss_fn = binary_cross_entropy
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
        self.a[0] = X.T  
        for i in range(self.depth):
            self.z[i+1] = np.dot(self.W[i], self.a[i]) + self.b[i]
            self.a[i+1] = self.activation_fn(self.z[i+1])
    #
    def _backprop(self, y):
        m = y.shape[0]  
        y = y.reshape(-1, m)
        delta = self.a[-1] - y
        for i in reversed(range(self.depth)):
            self.dW[i] = np.dot(delta, self.a[i].T) / m   
            self.db[i] = np.sum(delta, axis=1, keepdims=True) / m
            if i != 0:
                delta = np.dot(self.W[i].T, delta) * self.activation_dfn(self.z[i])
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
        n_samples = y.shape[0]
        epoch_losses = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            indices =[ i for i in range(n_samples)]
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for start_idx in range(0, n_samples, batch_size):
                if start_idx + batch_size > n_samples:
                    end_idx = n_samples
                else:
                    end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                self._feedforward(X_batch)
                self._backprop(y_batch)
                self._update_params(lr)
            compute_loss = self._compute_loss(X, y)
            epoch_losses.append(compute_loss)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Сompute_loss: {compute_loss}")

        return epoch_losses
    
    #
    def predict(self, X):
        a = X.T
        for i in range(self.depth):
            a = np.dot(self.W[i], a) + self.b[i]
            a = self.activation_fn(a)
        return a.T  
    

