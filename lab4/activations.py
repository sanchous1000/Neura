import numpy as np
import torch

class Relu_torch:
    def __init__(self):
        self.X = None

    def forward(self, X):
        """
        Прямой проход для функции активации ReLU.

        :param X: Входной тензор.
        :return: Результат применения ReLU.
        """
        self.X = X
        return torch.maximum(X, torch.tensor(0.0, device=X.device))

    def backward(self, dout):
        """
        Обратный проход для функции активации ReLU.

        :param dout: Градиенты от следующего слоя.
        :return: Градиенты относительно входа.
        """
        dx = dout.clone()
        dx[self.X <= 0] = 0
        return dx

class LeakyRelu_torch:
    def __init__(self, alpha=0.2):
        """
        Конструктор для функции активации LeakyReLU.

        :param alpha: Коэффициент утечки (для отрицательных значений).
        """
        self.alpha = alpha
        self.X = None

    def forward(self, X):
        """
        Прямой проход для функции активации LeakyReLU.

        :param X: Входной тензор.
        :return: Результат применения LeakyReLU.
        """
        self.X = X
        return torch.where(X > 0, X, self.alpha * X)

    def backward(self, dout):
        """
        Обратный проход для функции активации LeakyReLU.

        :param dout: Градиенты от следующего слоя.
        :return: Градиенты относительно входа.
        """
        dx = dout.clone()
        dx[self.X > 0] = 1
        dx[self.X <= 0] = self.alpha
        return dx

import torch

class Sigmoid_t:
    def __init__(self, device='cuda'):
        """
        Инициализация класса с поддержкой устройства (CPU или CUDA).
        Если устройство не указано, выбирается автоматически.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        """
        Прямое вычисление сигмоиды.
        """
        x = x.to(self.device)  # Перенос тензора на выбранное устройство
        x = torch.clamp(x, 1e-15, 1 - 1e-15)  # Ограничение значений для стабильности
        return 1 / (1 + torch.exp(-x))
    
    def backward(self, x):
        """
        Вычисление производной сигмоиды.
        """
        x = x.to(self.device)  # Перенос тензора на выбранное устройство
        sig = self.forward(x)  # Вычисление сигмоиды
        return sig * (1 - sig)




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
    def backward_(self,dout):
        dx = dout.copy()
        dx[self.X <= 0] = 0
        return dx
    


class LeakyRelu:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.X = None
    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, self.alpha * X)
    def backward(self, dout):
        dx = dout.copy()
        dx[self.X > 0] = 1  
        dx[self.X <= 0] = self.alpha  
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
    def backward_(self, X):
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
    def backward_(self, x):
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


