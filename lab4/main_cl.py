import numpy as np
rng = np.random.default_rng(51)



# Модели
class LeakyRelu:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.X = None

    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, dout):
        dx = dout * np.where(self.X > 0, 1, self.alpha)
        return dx

class Encoder:
    def __init__(self, input_size=784, hidden_dims=(256, 128), z_dim=50):
        self.fc1 = Fullyconnected(input_size, hidden_dims[0])
        self.fc2 = Fullyconnected(hidden_dims[0], hidden_dims[1])
        self.mu = Fullyconnected(hidden_dims[1], z_dim)
        self.log_var = Fullyconnected(hidden_dims[1], z_dim)
        self.activation = LeakyRelu()

    def forward(self, x):
        h = self.activation.forward(self.fc1.forward(x))
        h = self.activation.forward(self.fc2.forward(h))
        mu = self.mu.forward(h)
        log_var = self.log_var.forward(h)
        return mu, log_var

    def backward(self, dmu, dlogvar):
        dh = self.mu.backward(dmu) + self.log_var.backward(dlogvar)
        dh = self.fc2.backward(dh)
        self.fc1.backward(dh)

class Fullyconnected: 
    def __init__(self, input_size, output_size, derivative = False ):
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
    
    def backward_(self, dout):
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
