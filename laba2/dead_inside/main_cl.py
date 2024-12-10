import numpy as np 
from activations import Sigmoid
from adder import col2im_B, col2im, im2col


rng = np.random.default_rng(51)



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

    def backprop(self, dout):
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
    

class Conv:
    def __init__(self,amount_filters: int, height: int, width: int, chanels_amount: int = 1, padding: int = 0, stride:int = 1):
        self.w = self._init_filter_weights((chanels_amount, height, width), amount_filters)
        self.bias = self._init_filter_bias(amount_filters)
        self.height = height
        self.width = width
        self.padding = padding
        self.stride = stride
        self.im2col = im2col
        self.col2im = col2im
        self.t = 1
        self.mW = np.zeros_like(self.w) 
        self.mb = np.zeros_like(self.bias) 
        self.vW = np.zeros_like(self.w) 
        self.vb = np.zeros_like(self.bias) 
        
    def _init_filter_bias(self, amount_filters ):
        return np.zeros((amount_filters,1)) 

    def _init_filter_weights(self, arch, amount_filters ):
        channels_amount = arch[0]
        height = arch[1]
        width = arch[2]
        limit = np.sqrt(6. / (channels_amount * height * width + channels_amount * height * width))
        return rng.uniform(-limit, limit + 1e-5, size=(amount_filters, channels_amount, height, width))
    
    def forward(self, x):
        self.pad_num = self.padding
        F, C, HH, WW = self.w.shape
        N,C,H,W = x.shape
        H_prime = (H+2*self.pad_num-HH) // self.stride + 1
        W_prime = (W+2*self.pad_num-WW) // self.stride + 1
        out = None
        self.im_pad = np.pad(x,((0,0),(0,0),(self.pad_num,self.pad_num),(self.pad_num,self.pad_num)),'constant')
        self.im_col = im2col(self.im_pad,HH,WW,self.stride)
        self.filter_col = np.reshape(self.w,(F,-1))
        conv_it = np.dot(self.im_col, self.filter_col.T) + self.bias.T
        out = col2im(conv_it,H_prime,W_prime)
        self.x = x
        return out
    
    def backward(self, dout):
        N,Cx,H,W = self.x.shape
        F, C, HH, WW = self.w.shape
        dout_resh = np.reshape(dout,(F,-1))
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, N))
        dout = np.concatenate(dout, axis=-1)
        self.db = np.sum(dout_resh, axis=1, keepdims=True).reshape(self.bias.shape) 
        self.dw = np.dot(dout, self.im_col.reshape(dout.shape[1],self.im_col.shape[2])).reshape(self.w.shape) 
        dx  = np.dot(self.filter_col.T,dout)
        dx = col2im_B(dx,self.x.shape, self.height, self.width, self.stride, self.pad_num)
        return dx
    
    def _update_params(self, lr=0.0000001, beta_1=0.9, beta_2=0.999, eps=1e-08): 
        self.mW  = beta_1*self.mW  + (1-beta_1)*self.dw
        self.mb  = beta_1*self.mb  + (1-beta_1)*self.db 
        self.vW  = beta_2*self.vW  + (1-beta_2)*(self.dw **2)
        self.vb  = beta_2*self.vb  + (1-beta_2)*(self.db **2)
        mW_corr = self.mW  / (1-beta_1**self.t)
        mb_corr = self.mb  / (1-beta_1**self.t)
        vW_corr = self.vW  / (1-beta_2**self.t)
        vb_corr = self.vb  / (1-beta_2**self.t)
        self.w  -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
        self.bias  -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        self.t += 1



class MaxPooling:
    def __init__(self,height: int, width: int, stride: int = 2):
        self.height = height
        self.width = width
        self.stride = stride
        
    def forward(self,x):
        self.x = x
        batch, F,  H, W = x.shape
        out_h = (H - self.height) // self.stride + 1
        out_w = (W - self.width) // self.stride + 1
        out = np.zeros((batch, F, out_h, out_w))
        self.mask = np.zeros(x.shape)
        for y in range(out_h):
            for x_i in range(out_w):
                window = x[:, :, 
                           y * self.stride : y * self.stride + self.height, 
                           x_i * self.stride : x_i * self.stride + self.width]
                patch = window.max(axis=(2,3))
                out[:, :, y, x_i] = patch
                mask = (window == patch[:, :, None, None])
                self.mask[:, :, y * self.stride : y * self.stride + self.height, x_i * self.stride : x_i * self.stride + self.width] += mask
        return out
    
    def backward(self, dout):
        batch, F, H, W = self.x.shape  
        pool_h = self.height
        pool_w = self.width
        stride = self.stride
        out_h = (H - pool_h) // stride + 1
        out_w = (W - pool_w) // stride + 1
        dX = np.zeros_like(self.x)
        for y in range(out_h):
            for x_i in range(out_w):
                window_mask = self.mask[:, :, 
                                         y * stride : y * stride + pool_h, 
                                         x_i * stride : x_i * stride + pool_w]
                dout_expanded = dout[:, :, y, x_i][:, :, None, None]
                dX[:, :, y * stride : y * stride + pool_h, 
                   x_i * stride : x_i * stride + pool_w] += window_mask * dout_expanded
        return dX
    
class Flatten:
    def __init__(self, size):
        self.size = size
    def forward(self, pool2_out):
        self.flatten_sh = pool2_out.shape
        flatten = pool2_out.reshape(pool2_out.shape[0], -1)
        return flatten
    def backward(self, dout):
        return dout.reshape(self.flatten_sh)
    

class AvgPool:
    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
    
        self.cache = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1
        X_col = im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0]//n_C, -1)
        A_pool = np.mean(X_col, axis=1)
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

    def backward(self, dout):
        X = self.cache
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        dout_flatten = dout.reshape(n_C, -1) / (self.f * self.f)
        dX_col = np.repeat(dout_flatten, self.f*self.f, axis=0)
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX


