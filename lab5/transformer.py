import numpy as np 
rng = np.random.default_rng(51)


import numpy as np

class LayerNorm:
    def __init__(self, input_dim, epsilon=1e-5):
        """
        Инициализация LayerNorm
        input_dim: размер входного вектора
        epsilon: маленькое значение для предотвращения деления на 0
        """
        self.gamma = np.ones((1, input_dim))  # Масштабирование
        self.beta = np.zeros((1, input_dim))  # Смещение
        self.epsilon = epsilon

    def forward(self, x):
        """
        Прямой проход
        x: входной тензор, размер (batch_size, input_dim)
        """
        self.x = x
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.variance = np.var(x, axis=1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.variance + self.epsilon)
        self.out = self.gamma * self.x_norm + self.beta
        return self.out

    def backward(self, dout):
        """
        Обратный проход
        dout: градиент по выходу слоя, размер (batch_size, input_dim)
        """
        batch_size, input_dim = self.x.shape

        # Градиенты по gamma и beta
        dgamma = np.sum(dout * self.x_norm, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        # Градиент по нормализованному входу
        dx_norm = dout * self.gamma

        # Градиенты по x
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=1, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.variance + self.epsilon), axis=1, keepdims=True) + \
                dvar * np.sum(-2 * (self.x - self.mean), axis=1, keepdims=True) / input_dim
        dx = dx_norm / np.sqrt(self.variance + self.epsilon) + \
             dvar * 2 * (self.x - self.mean) / input_dim + \
             dmean / input_dim

        return dx, dgamma, dbeta

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
class Fullyconnected: 
    def __init__(self, input_size, output_size, derivative = False, bias = True ):
        self.derivative = derivative
        #
        self.W = self._init_weights(input_size, output_size)
        self.b = self._init_biases(output_size)
        #opt_params
        self.t = 1
        self.mW =np.zeros_like(self.W) 
        self.mb = np.zeros_like(self.b) 
        self.vW =np.zeros_like(self.W) 
        self.vb = np.zeros_like(self.b) 

    def forward(self, X):
        self.a_l = X
        z = np.dot(X, self.W.T) + self.b
        return z 

    def backward(self, dout):
        m =  self.a_l.shape[0]
        self.dW = np.dot(dout.T, self.a_l) / m 
        self.db = np.sum(dout, axis = 0, keepdims=True) / m
        delta = np.dot(dout, self.W)
        return delta
    
    def update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08): 
        self.mW  = beta_1*self.mW  + (1-beta_1)*self.dW
        self.mb  = beta_1*self.mb  + (1-beta_1)*self.db 
        self.vW  = beta_2*self.vW  + (1-beta_2)*(self.dW **2)
        self.vb  = beta_2*self.vb  + (1-beta_2)*(self.db **2)
        mW_corr = self.mW  / (1-beta_1**self.t)
        mb_corr = self.mb  / (1-beta_1**self.t)
        vW_corr = self.vW  / (1-beta_2**self.t)
        vb_corr = self.vb  / (1-beta_2**self.t)
        self.W -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
        self.b  -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        self.t += 1
    def _init_weights(self, input_size, output_size):
        net_in = input_size
        net_out = output_size
        limit = np.sqrt(6. / (net_in + net_out))
        return rng.uniform(-limit, limit + 1e-5, size=(net_out, net_in)) 
    #
    def _init_biases(self, output_size):
        return np.zeros((1,output_size)) 


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        """
        Прямой проход для softmax.

        :param x: Входной массив, 2D-матрица (batch_size, num_classes)
        :return: Массив с вероятностями после применения softmax.
        """
        # Нормализация для избежания численной нестабильности
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dout):
        """
        Обратный проход для softmax.

        :param dout: Градиенты от следующего слоя
        :return: Градиенты относительно входа
        """
        batch_size, num_classes = self.output.shape
        dx = np.zeros_like(dout)
        
        # Jacobian матрица для softmax
        for i in range(batch_size):
            y = self.output[i]
            jacobian = np.diag(y) - np.outer(y, y)
            dx[i] = np.dot(jacobian, dout[i])
        
        return dx


class Head:
    def __init__(self, head_size, C ):
        self.head_size = head_size
        self.k = Fullyconnected(C , head_size)
        self.q = Fullyconnected(C , head_size)
        self.v = Fullyconnected(C , head_size)
        self.softmax = Softmax()
    
    def forward(self, X):
        B, T, C = X.shape
        key = self.k.forward(X)
        query = self.q.forward(X)
        
        value = self.v.forward(X)
        print(key.transpose(0, 2, 1).shape)
        self.w = np.matmul(query,key.transpose(0, 2, 1))

        output = np.matmul(self.softmax.forward(self.w), value) * self.head_size ** -0.5
        return output



class MultiHeadAttention:
    def __init__(self, number_heads,  head_size, n_emb ):
        self.heads = [Head(head_size=head_size) for _ in range(number_heads)]
        self.proj = Fullyconnected(n_emb, n_emb)
    def forward(self, X):
        out = np.concat([head.forward(X) for head in self.heads], axis=-1)
        out = self.proj.forward(out)
        return out

class Feedforward:
    def __init__(self,n_emb ):
        self.fc1 = Fullyconnected(n_emb, 4 * n_emb)
        self.relu = Relu()
        self.fc2 = Fullyconnected(4 * n_emb, n_emb)
        self.layers = [
            self.fc1,
            self.relu,
            self.fc2
        ]
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class Block:
    def __init__(self, n_emb, n_head):
        head_size = n_emb//n_head
        self.self_att = MultiHeadAttention(n_head, head_size)
        self.fd = Feedforward(n_emb)
        self.ln1 = LayerNorm(n_emb)
        self.ln2 = LayerNorm(n_emb)
    def forward(self, X):
        X = X + self.self_att.forward(self.ln1.forward(X))
        X = X + self.fd.forward(self.ln2.forward(X))





c = Head(16, 32)
r = c.forward(np.ones((4,8,32)))
print(r)
        