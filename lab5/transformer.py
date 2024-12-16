import numpy as np 
rng = np.random.default_rng(51)

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
    def __init__(self, number_heads,  head_size):
        self.heads = [Head(head_size=head_size) for _ in range(number_heads)]
    def forward(self, X):
        return np.concat([head.forward(X) for head in self.heads], axis=-1)



class Block:
    def __init__(self, n_emb, n_head):
        head_size = n_emb//n_head
        self.self_att = MultiHeadAttention(n_head, head_size)
        self.fd = Fu




c = Head(16, 32)
r = c.forward(np.ones((4,8,32)))
print(r)
        