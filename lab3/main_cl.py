import numpy as np
from activations import Tanh, Sigmoid
from tqdm import tqdm
rng = np.random.default_rng(51)


def MSE(dout):
    return np.mean(dout**2)



class RNN: 
    def __init__(self, input_size, output_size, hidden_size):
        self.derivative = Tanh().backward
        self.activation = Tanh().forward
       
        self.hidden_size = hidden_size
        self.output_size = output_size
        #
        self.compute_loss = MSE
        self.Wx = self._init_weights(input_size, hidden_size)
        self.Wy = self._init_weights(hidden_size, output_size)
        self.Wh = self._init_weights(hidden_size, hidden_size)
        #
        self.bh = self._init_biases(hidden_size)
        self.by = self._init_biases(output_size)
        #opt_params
        self.t = 1
        self.mWx = np.zeros_like(self.Wx)
        self.vWx = np.zeros_like(self.Wx)
        self.mWh = np.zeros_like(self.Wh)
        self.vWh = np.zeros_like(self.Wh)
        self.mWo = np.zeros_like(self.Wy)
        self.vWo = np.zeros_like(self.Wy)
        self.mbh = np.zeros_like(self.bh)
        self.vbh = np.zeros_like(self.bh)
        self.mbo = np.zeros_like(self.by)
        self.vbo = np.zeros_like(self.by)

    def _init_weights(self, input_size, output_size):
        net_in = input_size
        net_out = output_size
        limit = np.sqrt(6. / (net_in + net_out))
        return rng.uniform(-limit, limit + 1e-5, size=(net_in, net_out)) 
    #
    def _init_biases(self, output_size):
        return np.zeros((1,output_size)) 

    def forward(self, X):
        batch_size, N, inputs = X.shape
        self.input = X
        self.input_agg = np.zeros((batch_size, N, self.hidden_size ))
        self.h = np.zeros((batch_size,self.hidden_size))
        self.hidden = np.zeros((batch_size, N, self.hidden_size))
        self.output = np.zeros((batch_size, N, self.output_size))
        for n in range(N):
            x = X[... , n, :]
            h = np.dot(x, self.Wx) + self.bh + np.dot(self.h, self.Wh)
            self.h = self.activation(h)
            y = np.dot(self.h, self.Wy) + self.by 
            self.input_agg[... , n, :] = h
            self.hidden[... , n, :] = self.h
            self.output[... , n, :] = y    
        return self.output

    def backprop(self, dout):
        m, N, _ = dout.shape
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.dWy = np.zeros_like(self.Wy)
        self.dby = np.zeros_like(self.by)
        self.dbh = np.zeros_like(self.bh)

        dh_n = np.zeros(self.h.shape)
        for n in reversed(range(N)):
            delta = dout[: , n, :]
            self.dby += np.sum(delta, axis = 0, keepdims = True) / m
            self.dWy += np.dot(self.hidden[: , n, :].T, delta)

            dh = np.dot(delta, self.Wy.T) + dh_n

            dh *= self.derivative(self.input_agg[:, n, : ] if n > 0 else 0)

            dh_n = np.dot(dh, self.Wh.T)

            self.dWx +=  np.dot(self.input[: , n, :].T, dh)
            if n > 0:
                h_prev = self.hidden[: , n-1, :]
            else:
                h_prev = np.zeros_like(self.hidden[: , n, :])
                
            self.dWh +=  np.dot(h_prev.T, dh)
            self.dbh += np.sum(dh, axis=0, keepdims=True) 

    def train(self, X, y, epochs=1, batch_size=8, lr=1e-2):
        X = np.array(X) 
        y = np.array(y)
        n_samples = y.shape[0]
        epoch_losses = []
        loss = 0
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
                y_pred = self.forward(X_batch)
                delta = (y_pred - y_batch) / y_batch.size
                loss =  self.compute_loss(delta)
                self.backprop(delta)
                self.update_params(lr)
            epoch_losses.append(loss)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Сompute_loss: {loss}")
        return epoch_losses
    
    def update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08): 
        self.mWx = beta_1 * self.mWx + (1 - beta_1) * self.dWx
        self.vWx = beta_2 * self.vWx + (1 - beta_2) * (self.dWx ** 2)
        mWx_corr = self.mWx / (1 - beta_1 ** self.t)
        vWx_corr = self.vWx / (1 - beta_2 ** self.t)
        self.Wx -= lr * mWx_corr / (np.sqrt(vWx_corr) + eps)

      
        self.mWh = beta_1 * self.mWh + (1 - beta_1) * self.dWh
        self.vWh = beta_2 * self.vWh + (1 - beta_2) * (self.dWh ** 2)
        mWh_corr = self.mWh / (1 - beta_1 ** self.t)
        vWh_corr = self.vWh / (1 - beta_2 ** self.t)
        self.Wh -= lr * mWh_corr / (np.sqrt(vWh_corr) + eps)

        
        self.mWo = beta_1 * self.mWo + (1 - beta_1) * self.dWy
        self.vWo = beta_2 * self.vWo + (1 - beta_2) * (self.dWy ** 2)
        mWo_corr = self.mWo / (1 - beta_1 ** self.t)
        vWo_corr = self.vWo / (1 - beta_2 ** self.t)
        self.Wy -= lr * mWo_corr / (np.sqrt(vWo_corr) + eps)

        
        self.mbh = beta_1 * self.mbh + (1 - beta_1) * self.dbh
        self.vbh = beta_2 * self.vbh + (1 - beta_2) * (self.dbh ** 2)
        mbh_corr = self.mbh / (1 - beta_1 ** self.t)
        vbh_corr = self.vbh / (1 - beta_2 ** self.t)
        self.bh -= lr * mbh_corr / (np.sqrt(vbh_corr) + eps)

        
        self.mbo = beta_1 * self.mbo + (1 - beta_1) * self.dby
        self.vbo = beta_2 * self.vbo + (1 - beta_2) * (self.dby ** 2)
        mbo_corr = self.mbo / (1 - beta_1 ** self.t)
        vbo_corr = self.vbo / (1 - beta_2 ** self.t)
        self.by -= lr * mbo_corr / (np.sqrt(vbo_corr) + eps)
        self.t += 1
    
    


class LSTM:
    def __init__(self, input_size, output_size, hidden_size):
        self.derivative_tan = Tanh().backward
        self.activation_tan = Tanh().forward
        self.derivative_sig = Sigmoid().backward
        self.activation_sig = Sigmoid().forward
       
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        #
        self.compute_loss = MSE
        self.Wx_f = self._init_weights(input_size, hidden_size)
        self.Wx_c = self._init_weights(input_size, hidden_size)
        self.Wx_o = self._init_weights(input_size, hidden_size)
        #
        self.Wh_f = self._init_weights(hidden_size, hidden_size)
        self.Wh_c = self._init_weights( hidden_size, hidden_size)
        self.Wh_o = self._init_weights( hidden_size, hidden_size)
        #
        self.Wh_y = self._init_weights(hidden_size, output_size)
        #
        self.bf = self._init_biases(hidden_size)
        self.bc = self._init_biases(hidden_size)
        self.bo = self._init_biases(hidden_size)
        self.by = self._init_biases(output_size)

    def _init_weights(self, input_size, output_size):
        net_in = input_size
        net_out = output_size
        limit = np.sqrt(6. / (net_in + net_out))
        return rng.uniform(-limit, limit + 1e-5, size=(net_in, net_out)) 
    #
    def _init_biases(self, output_size):
        return np.zeros((1,output_size)) 
    

    def forward(self, X):
        batch_size, N, inputs = X.shape
        self.input = X
        self.x_h_concat = np.zeros((batch_size, N, self.hidden_size ))
        self.h = np.zeros((batch_size,self.hidden_size))
        self.c = np.zeros((batch_size,self.hidden_size))
        self.f = np.zeros((batch_size, N , self.hidden_size))
        self.c_p = np.zeros((batch_size, N , self.hidden_size))
        self.f_p = np.zeros((batch_size, N , self.hidden_size))
        self.o_p = np.zeros((batch_size, N , self.hidden_size))
        self.c_up = np.zeros((batch_size, N , self.hidden_size))
        self.o = np.zeros((batch_size, N , self.hidden_size))
        self.hidden = np.zeros((batch_size, N, self.hidden_size))
        self.output = np.zeros((batch_size, N, self.output_size))
        c_p = 0
        for n in range(N):
            x = X[... , n, :]
            f_p= np.dot(x, self.Wx_f) + np.dot(self.h, self.Wh_f) + self.bf
            f = self.activation_sig(f_p)
            c_p = np.dot(x, self.Wx_c) + np.dot(self.h, self.Wh_c) + self.bc
            self.c_ = self.activation_tan(c_p)
            o_p = np.dot(x, self.Wx_o) + np.dot(self.h, self.Wh_o) + self.bo
            o = self.activation_sig(o_p)

            c_up = f * c_p + (1-f)*self.c_

            self.h = o * self.activation_tan(c_up)
            y = np.dot(self.h, self.Wh_y) + self.by 
            
            self.output[... , n, :] = y
            self.f[... , n, :] = f
            self.o[... , n, :] = o 
            self.c[... , n, :] = self.c_

            self.c_up[:, n, :] = c_up

            self.f_p[... , n, :] = f
            self.o_p[... , n, :] = o_p
            self.c_p[... , n, :] = c_p 

            self.hidden[... , n, :] = self.h
        return self.output
    
    def backprop(self, dout):
        m, N, _ = dout.shape
        self.dWx_f = np.zeros_like(self.Wx_f)
        self.dWx_c = np.zeros_like(self.Wx_c)
        self.dWx_o = np.zeros_like(self.Wx_o)
        #
        self.dWh_f = np.zeros_like( self.Wh_f)
        self.dWh_c = np.zeros_like(self.Wh_c)
        self.dWh_o = np.zeros_like(self.Wh_o)
        #
        self.dWh_y = np.zeros_like(self.Wh_y)
        #
        self.dbf = np.zeros_like(self.bf)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        self.dby = np.zeros_like( self.by)
        #
       
        dc_n = np.zeros((m, self.hidden_size))

        dh_n = np.zeros(self.h.shape)
        for n in reversed(range(N)):
            delta = dout[: , n, :]
            if n > 0:
                h_prev = self.hidden[: , n-1, :]
                c_prev = self.c_p[: , n-1, :]
            else:
                h_prev = np.zeros_like(self.hidden[: , n, :])
                c_prev = np.zeros_like(self.c_p[: , n, :])
            
            self.dby += np.sum(delta, axis = 0, keepdims = True) / m
            self.dWh_y += np.dot(self.hidden[: , n, :].T, delta)

            dh = np.dot(delta, self.Wh_y.T) + dh_n
            dc = dh * self.o[:, n, :] * self.derivative_tan(self.c_up[: , n, :]) + dc_n
            df = dc * (c_prev - self.c[:, n , :] ) 
            dc_prev = dc * self.f[:, n, :]
            dc_ = dc * (1 - self.f[:, n , :])
            df = df * self.derivative_tan(self.f[:, n , :])
            do = dh * self.activation_tan(self.c_up[:, n , :]) * self.derivative_sig(self.o[:, n, :])
            dc_tilde_p = dc_ * (1 - self.c[:, n, :] ** 2)
    


            
            self.dWh_f += np.dot(h_prev.T, df)
            self.dbf += np.sum(df, axis=0)

            self.dWx_o += np.dot(self.input[:, n, :].T, do)
            self.dWh_o += np.dot(h_prev.T, do)
            self.dbo += np.sum(do, axis=0)

            self.dWx_c += np.dot(self.input[:, n, :].T, dc_tilde_p)
            self.dWh_c += np.dot(h_prev.T, dc_tilde_p)
            self.dbc += np.sum(dc_tilde_p, axis=0)

            
            dh_n = np.dot(df, self.Wh_u.T) + np.dot(do, self.Wh_o.T) + \
                      np.dot(dc_tilde_p, self.Wh_c.T)
            dc_n = dc_prev



    

    
