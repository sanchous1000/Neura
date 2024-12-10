import numpy as np
class Relu:
    def __init__(self):
        self.X = None
    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)
    def backward(self,dout):
        dx = dout.copy()
        dx[self.X <= 0] = 0
        return dx
    
class Tanh:
    def __init__(self):
        pass
    def forward(self, X, a = 1.72):
        self.a = a
        self.X = X
        return self.a * np.tanh(X)
    def backward(self, X):
        return X *  (1-np.tanh(self.X)**2)   

class Sigmoid:
    def __init__(self):
        pass
    def forward(self,x):
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return 1 / (1 + np.exp(-x))
    def backward(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)

class Softmax:
    def __init__(self):
        pass
    def forward(self,x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True) 
    def backward(self, y_pred, y_true):
        return y_pred - y_true
    
def categorical_cross_entropy(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred))


