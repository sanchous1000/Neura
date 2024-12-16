import torch
import torch.nn as nn

class FullyConnected:
    def __init__(self, input_size, output_size, derivative=False, device=None):
        """
        Полносвязный слой, реализованный с использованием PyTorch.

        :param input_size: Размер входного слоя.
        :param output_size: Размер выходного слоя.
        :param derivative: Флаг для расчета производных (не используется в PyTorch).
        :param device: Устройство для вычислений (CPU или CUDA).
        """
        self.derivative = derivative
        self.device = device if device else torch.device('cpu')
        self.W = self._init_weights(input_size, output_size).to(self.device)
        self.b = self._init_biases(output_size).to(self.device)

        # opt_params
        self.t = 1
        self.mW = torch.zeros_like(self.W, device=self.device)
        self.mb = torch.zeros_like(self.b, device=self.device)
        self.vW = torch.zeros_like(self.W, device=self.device)
        self.vb = torch.zeros_like(self.b, device=self.device)

    def forward(self, X):
        """
        Прямой проход.

        :param X: Входные данные (батч).
        :return: Результат линейной трансформации.
        """
        self.a_l = X.to(self.device)
        z = torch.matmul(self.a_l, self.W.T) + self.b
        return z

    def backward(self, dout):
        """
        Обратный проход.

        :param dout: Градиенты от следующего слоя.
        :return: Градиенты относительно входа.
        """
        dout = dout.to(self.device)
        m = self.a_l.shape[0]
        self.dW = torch.matmul(dout.T, self.a_l) / m
        self.db = torch.sum(dout, dim=0, keepdim=True) / m
        delta = torch.matmul(dout, self.W)
        return delta

    def update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08):
        """
        Обновление параметров с использованием Adam.

        :param lr: Скорость обучения.
        :param beta_1: Коэффициент для первого момента (0.9 по умолчанию).
        :param beta_2: Коэффициент для второго момента (0.999 по умолчанию).
        :param eps: Малое значение для численной стабильности (1e-08 по умолчанию).
        """
        self.mW = beta_1 * self.mW + (1 - beta_1) * self.dW
        self.mb = beta_1 * self.mb + (1 - beta_1) * self.db
        self.vW = beta_2 * self.vW + (1 - beta_2) * (self.dW ** 2)
        self.vb = beta_2 * self.vb + (1 - beta_2) * (self.db ** 2)

        mW_corr = self.mW / (1 - beta_1 ** self.t)
        mb_corr = self.mb / (1 - beta_1 ** self.t)
        vW_corr = self.vW / (1 - beta_2 ** self.t)
        vb_corr = self.vb / (1 - beta_2 ** self.t)

        self.W -= lr * mW_corr / (torch.sqrt(vW_corr) + eps)
        self.b -= lr * mb_corr / (torch.sqrt(vb_corr) + eps)

        self.t += 1

    def _init_weights(self, input_size, output_size):
        """
        Инициализация весов.

        :param input_size: Размер входного слоя.
        :param output_size: Размер выходного слоя.
        :return: Инициализированные веса.
        """
        limit = (6.0 / (input_size + output_size)) ** 0.5
        return torch.empty(output_size, input_size).uniform_(-limit, limit)

    def _init_biases(self, output_size):
        """
        Инициализация смещений (bias).

        :param output_size: Размер выходного слоя.
        :return: Инициализированные смещения.
        """
        return torch.zeros(1, output_size)
    

