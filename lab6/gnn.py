import numpy as np
rng = np.random.default_rng(51)

class Relu:
    def __init__(self):
        self.X = None
    def forward(self, X, A = 0):
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
        self.db = np.sum(dout, axis=0, keepdims=True) / m
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


class GraphConv(Fullyconnected):
    def forward(self, X, A):
        self.A = A
        self.DaD = self.dad(A)  
        self.X = np.dot(self.DaD, X)  
        self.Z = super().forward(self.X)  
        return self.Z

    def dad(self, A):
        Da = A.copy()
        np.fill_diagonal(Da, Da.diagonal() + 1.0) 
        D = np.sum(Da, axis=1) + 1e-9  
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D))  
        return D_inv_sqrt @ Da @ D_inv_sqrt

    def backward(self, dout):
        delta = super().backward(dout)  
        dX = np.dot(self.DaD.T, delta) 
        return dX
    def update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08):
        super().update_params(lr=lr, beta_1=beta_1, beta_2=beta_2, eps=eps)

    

class Gap(Fullyconnected):
    def __init__(self, input_size, output_size, bias=True):
        super().__init__(input_size, output_size, bias=bias)

    def forward(self, Z, A = 0):
        self.Z = Z
        self.h = np.mean(Z, axis=0)  
        y_pred = super().forward(self.h[None, :])[0]  
        return y_pred

    def backward(self, dout):
        N = self.Z.shape[0]
        dh = super().backward(dout[None, :])[0] 
        dZ = np.ones_like(self.Z) * (dh / N)
        return dZ
    
    def update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08):
        super().update_params(lr=lr, beta_1=beta_1, beta_2=beta_2, eps=eps)

    
class GNN:
    def __init__(self,layers):
        self.layers = layers

    def forward(self, X, A = 0):
        for layer in self.layers:
            X = layer.forward(X, A)
        return X
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def update_params(self, lr = 0.001):
        for layer in self.layers:
            if 'Relu' not in str(layer.__class__):
                if 'Flatten' not in str(layer.__class__):
                    layer.update_params(lr)


