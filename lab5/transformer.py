
import numpy as np 
  

rng = np.random.default_rng(51)

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

class LayerNorm:
    def __init__(self, input_dim, epsilon=1e-5):
        self.input_dim = input_dim
        self.gamma = np.ones((input_dim,))  # Масштабирование
        self.beta = np.zeros((input_dim,))  # Смещение
        self.epsilon = epsilon
        
        # Параметры для оптимизации
        self.t = 1
        self.mW = np.zeros_like(self.gamma) 
        self.mb = np.zeros_like(self.beta) 
        self.vW = np.zeros_like(self.gamma) 
        self.vb = np.zeros_like(self.beta) 

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.variance + self.epsilon)
        self.out = self.gamma * self.x_norm + self.beta
        return self.out

    def backward(self, dout):
       
        self.dW = np.sum(dout * self.x_norm, axis=0)
        self.db = np.sum(dout, axis=0)
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.variance + self.epsilon), axis=-1, keepdims=True) + \
                dvar * np.sum(-2 * (self.x - self.mean), axis=-1, keepdims=True) / self.input_dim
        dx = dx_norm / np.sqrt(self.variance + self.epsilon) + \
             dvar * 2 * (self.x - self.mean) / self.input_dim + \
             dmean / self.input_dim

        return dx

    def update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08): 
        dW_agg = np.sum(self.dW, axis=0) if self.dW.ndim > 1 else self.dW
        db_agg = np.sum(self.db, axis=0) if self.db.ndim > 1 else self.db
        self.mW = beta_1 * self.mW + (1 - beta_1) * dW_agg
        self.mb = beta_1 * self.mb + (1 - beta_1) * db_agg
        self.vW = beta_2 * self.vW + (1 - beta_2) * (dW_agg ** 2)
        self.vb = beta_2 * self.vb + (1 - beta_2) * (db_agg ** 2)
        mW_corr = self.mW / (1 - beta_1**self.t)
        mb_corr = self.mb / (1 - beta_1**self.t)
        vW_corr = self.vW / (1 - beta_2**self.t)
        vb_corr = self.vb / (1 - beta_2**self.t)
        self.gamma -= lr * mW_corr / (np.sqrt(vW_corr) + eps)
        self.beta -= lr * mb_corr / (np.sqrt(vb_corr) + eps)
        self.t += 1

    


class Fullyconnected: 
    def __init__(self, input_size, output_size, type = 'transform', derivative = False, bias = True ):
        self.derivative = derivative
        self.type = type
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
        if self.type == 'transform':
            self.dW =  np.sum(np.matmul(dout.transpose(0, 2, 1), self.a_l), axis = 0)
            self.db = np.sum(dout, axis = (0, 1))
        else:
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

    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        self.output = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
        return self.output

    def backward(self, dOut):
        dX = self.output * (dOut - np.sum(dOut * self.output, axis=-1, keepdims=True))
        return dX


class Head:
    def __init__(self, head_size, C ):
        self.head_size = head_size
        self.k = Fullyconnected(C , head_size)
        self.q = Fullyconnected(C , head_size)
        self.v = Fullyconnected(C , head_size)
        self.softmax = Softmax()
    
    def forward(self, X):
        self.key = self.k.forward(X)
        self.query = self.q.forward(X)
        self.value = self.v.forward(X)
        self.w = np.matmul(self.query, self.key.transpose(0, 2, 1))
        self.ws = self.softmax.forward(self.w)
        output = np.matmul(self.ws, self.value) * self.head_size ** -0.5
        return output
    
    def backward(self, dOut):
        dOut = dOut * (self.head_size ** -0.5)
        dValue = np.matmul(self.ws.transpose(0, 2, 1), dOut)
        dW_softmax = np.matmul(dOut, self.value.transpose(0, 2, 1))
        dW = self.softmax.backward(dW_softmax)
        dQuery = np.matmul(dW, self.key)
        dKey = np.matmul(dW.transpose(0, 2, 1), self.query)
        dX_v = self.v.backward(dValue)
        dX_q = self.q.backward(dQuery)
        dX_k = self.k.backward(dKey)
        dX = dX_v + dX_q + dX_k
        return dX
    
    def update_params(self, lr = 1e-6):
        self.k.update_params(lr)
        self.q.update_params(lr)
        self.v.update_params(lr)








class MultiHeadAttention:
    def __init__(self, number_heads,  head_size, n_emb ):
        self.heads = [Head(head_size=head_size, C = n_emb) for _ in range(number_heads)]
        self.proj = Fullyconnected(n_emb, n_emb)
    def forward(self, X):
        out = np.concatenate([head.forward(X) for head in self.heads], axis=-1)
        out = self.proj.forward(out)
        return out
    
    def backward(self, dOut):
        dConcat = self.proj.backward(dOut)
        dX_heads = []
        split_dOut = np.split(dConcat, len(self.heads), axis=-1)
        for i, head in enumerate(self.heads):
            dX_heads.append(head.backward(split_dOut[i]))
        dX = np.sum(np.stack(dX_heads), axis=0)
        return dX
    def update_params(self, lr = 1e-6):
        self.proj.update_params(lr) 
    

    

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
    def update_params(self,  lr = 1e-6 ):
        for layer in reversed(self.layers):
            try:
                layer.update_params(lr)
            except:
                pass



    
class Meaner:
    def __init__(self):
        pass 
    def forward(self, X):
        self.time = X.shape[1]
        return np.mean(X, axis = 1)
    def backward(self, dout):
        din = dout[:, np.newaxis, :] / self.time  # Расширяем размерность вдоль оси tim
        return np.repeat(din, self.time, axis=1) 
    
class Classification:
    def __init__(self,input,output ):
        self.meaner = Meaner()
        self.fc1 = Fullyconnected(input, output, type = '34')
        self.sm = Softmax()
        self.layers = [
            self.meaner,
            self.fc1,
             self.sm
        ]
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    def update_params(self,  lr = 1e-6 ):
        for layer in reversed(self.layers):
            try:
                layer.update_params(lr)
            except:
                pass

    



class Block_encoder:
    def __init__(self, n_emb, n_head):
        head_size = n_emb//n_head
        self.self_att = MultiHeadAttention(n_head, head_size, n_emb)
        self.fd = Feedforward(n_emb)
        self.ln1 = LayerNorm(n_emb)
        self.ln2 = LayerNorm(n_emb)
        
    def forward(self, X):
        self.X = X 
        self.self_att_out = self.self_att.forward(X)
        self.ln1_out = X + self.ln1.forward(self.self_att_out )
        self.fd_out = self.fd.forward(self.ln1_out)
        self.ln2_out = self.ln1_out + self.ln2.forward(self.fd_out)

        return  self.ln2_out

    def backward(self, dout):
        dout_ln2 = self.ln2.backward(dout)
        dout_fd = dout + self.fd.backward(dout_ln2)
        dout_ln1 = self.ln1.backward(dout_fd)
        dout_self_att = self.self_att.backward(dout_ln1)

        return dout_fd +  dout_self_att
    def update_params(self,  lr = 1e-6 ):
        self.ln1.update_params(lr)
        self.ln2.update_params(lr)
        self.fd.update_params(lr)
        self.self_att.update_params(lr)

    
 
 

class SentimentEncoder:
    def __init__(self, n_emb, n_head, output_classes):
        self.encoder = Block_encoder(n_emb=n_emb, n_head=n_head)
        self.classifier = Classification(n_emb, output_classes)
    
    def forward(self, X):
        enc_output = self.encoder.forward(X)
        logits = self.classifier.forward(enc_output)
        return logits
    
    def backward(self, logits, y):
        loss_grad = logits - y
        grad_classifier = self.classifier.backward(loss_grad)
        self.encoder.backward(grad_classifier)

    def update_params(self,  lr = 1e-6 ):
        self.encoder.update_params(lr)
        self.classifier.update_params(lr)
    
    
