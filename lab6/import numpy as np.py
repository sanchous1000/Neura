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
        try:
            self.dW = np.dot(dout.transpose(0, 2, 1), self.a_l) / m 
        except:
            self.dW = np.dot(dout.T, self.a_l) / m 

        self.db = np.sum(dout, axis = 1) / m
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


class GrathConv(Fullyconnected):
    def __call__(self):
        super().__call__()
    def forward(self, X, A):
        DaD = self.dad(A)
        print(DaD)
        print(X)
        self.X = DaD @ X
        return super().forward(self.X)


    def dad(self, A):
        Da = A.copy()
        I = Da.diagonal() + 1.0
        np.fill_diagonal(Da, I)
        self.D = np.sum(Da, axis=1) + 1e-9
        self.D = self.D ** (-1/2)
        return (Da *  self.D[:, None]) *  self.D[None, :]
    


