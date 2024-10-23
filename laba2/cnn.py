import numpy as np
from MLP import MLP
rng = np.random.default_rng(51)

def relu(x):
    return np.maximum(0, x)

def col2im_back(dim_col, h_prime, w_prime, stride, hh, ww, C):
    """
    Обратное преобразование из колоночного представления в изображение.

    Parameters:
    - dim_col: Массив формы (batch * F, H' * W', C * hh * ww)
    - h_prime, w_prime: Размеры выходной карты после пуллинга
    - stride: Шаг пуллинга
    - hh, ww: Высота и ширина окна пуллинга
    - C: Количество каналов

    Returns:
    - dx: Градиент относительно входного изображения, форма (batch * F, C, H, W)
    """
    batchF = dim_col.shape[0]
    dx = np.zeros((batchF, C, (h_prime - 1) * stride + hh, (w_prime - 1) * stride + ww))
    
    for i in range(h_prime * w_prime):
        row = dim_col[:, i, :]  # (batchF, C * hh * ww)
        h_start = (i // w_prime) * stride
        w_start = (i % w_prime) * stride
        dx[:, :, h_start:h_start + hh, w_start:w_start + ww] += row.reshape(batchF, C, hh, ww)
    
    return dx

def im2col(x,hh,ww,stride):
    batch, C, h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros((batch, new_h*new_w,hh*ww*C))
    for i in range(new_h):
       for j in range(new_w):
            patch = x[:,:, i*stride:i*stride+hh,j*stride:j*stride+ww]
            col[:,i*new_w+j,:] = np.reshape(patch,(batch,-1))
    return col

def col2im(mul,h_prime,w_prime):
    batch = mul.shape[0]
    F = mul.shape[-1]
    out = mul.reshape((batch, F, h_prime, w_prime))
    return out

class Conv:
    def __init__(self,amount_filters: int, height: int, width: int,chanels_amount: int = 1):        
        self.w = self._init_filter_weights((chanels_amount, height, width), amount_filters)
        self.bias = self._init_filter_bias(amount_filters)
        self.height = height
        self.width = width
        self.im2col = im2col
        self.col2im = col2im
        
    def _init_filter_bias(self, amount_filters ):
        return np.zeros((1,amount_filters)) 

    def _init_filter_weights(self, arch, amount_filters ):
        channels_amount = arch[0]
        height = arch[1]
        width = arch[2]
        limit = np.sqrt(6. / (channels_amount * height * width + channels_amount * height * width))
        return rng.uniform(-limit, limit + 1e-5, size=(amount_filters, channels_amount, height, width))
    
    def conv_forward(self, x, conv_param):
        self.pad_num = conv_param['pad']
        self.stride = conv_param['stride']
        F, C, HH, WW = self.w.shape
        if len(x.shape) == 3:
            N,H,W = x.shape
            C = 1
            x = x.reshape((N,C,H,W))
        else:
            N,C,H,W = x.shape
        H_prime = (H+2*self.pad_num-HH) // self.stride + 1
        W_prime = (W+2*self.pad_num-WW) // self.stride + 1
        out = None
        im_pad = np.pad(x,((0,0),(0,0),(self.pad_num,self.pad_num),(self.pad_num,self.pad_num)),'constant')
        print(im_pad.shape)
        im_col = im2col(im_pad,HH,WW,self.stride)
        filter_col = np.reshape(self.w,(F,-1))
        conv_it = np.dot(im_col, filter_col.T) + self.bias
        out = col2im(conv_it,H_prime,W_prime)
        self.mask = (x, filter_col, conv_param )
        return out
    def conv_backward(self, dout):
        x, im_pad, im_col, filter_col, conv_param, out = self.mask
        stride = self.stride
        pad = self.pad_num
        F, C, HH, WW = self.w.shape
        N, F_out, H_prime, W_prime = dout.shape

        dBias = np.sum(dout, axis=(0, 2, 3), keepdims=True) 

        
        dout_reshaped = dout.transpose(0,2,3,1).reshape(-1, F) 
    
        dW = dout_reshaped.T.dot(im_col.reshape(-1, F)).reshape(F, C, HH, WW) 

    
        dIm_col = dout_reshaped.dot(filter_col) 
        dIm_col = dIm_col.reshape(N, H_prime * W_prime, C * HH * WW) 
        
       
        dIm_pad = np.zeros_like(im_pad)
        for n in range(N):
            dIm_pad[n] += col2im_back(dIm_col[n].T, H_prime, W_prime, stride, HH, WW, C) 


        if pad > 0:
            dX = dIm_pad[:, :, pad:-pad, pad:-pad]
        else:
            dX = dIm_pad

    
        self.dW = dW
        self.dBias = dBias

        return dX

    def update_params(self, lr=0.01):
        self.w -= lr * self.dW
        self.bias -= lr * self.dBias

    
class Pooling:
    def __init__(self,height: int, width: int, stride: int = 2):
        self.height = height
        self.width = width
        self.stride = stride
    def max_pooling(self,x):
        self.type = 'max'
        batch, F,  H, W = x.shape
        H_prime = (H - self.height) // self.stride + 1
        W_prime = (W - self.width) // self.stride + 1
        x_reshaped = x.reshape(batch * F, 1, H, W)
        im_col = im2col(x_reshaped, self.height, self.width, self.stride)  
        out = np.max(im_col, axis=2) 
        out = out.reshape(batch, F, H_prime, W_prime)
        self.x = x
        return out
    
    def pool_backward(self, dout):
        batch, F, H, W = self.x.shape  
        pool_h = self.height
        pool_w = self.width
        stride = self.stride
        dx = np.zeros(self.x.shape)

       
        return None

    def average_pooling(self,x):
        self.type = 'average'
        batch, F,  H, W = x.shape
        out_h = (H - self.height) // self.stride + 1
        out_w = (W - self.width) // self.stride + 1
        out = np.zeros((batch, F, out_h, out_w))
        for y in range(out_h):
            for x_pos in range(out_w):
                patch = x[:,:, y*self.stride:y*self.stride+self.height, x_pos*self.stride:x_pos*self.stride+self.width].mean(axis=(2,3))
                out[:, :, y, x_pos] = patch
        return out
