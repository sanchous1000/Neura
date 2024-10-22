import numpy as np
from MLP import MLP
rng = np.random.default_rng(51)

def relu(x):
    return np.maximum(0, x)

def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c,H,W])
    for i in range(h_prime*w_prime):
        row = dim_col[i,:]
        h_start = (i / w_prime) * stride
        w_start = (i % w_prime) * stride
        dx[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
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
    out = np.zeros([batch,F,h_prime,w_prime])
    out[:,:,:,:] = np.reshape(mul,(batch, F,h_prime,w_prime))
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
        im_col = im2col(im_pad,HH,WW,self.stride)
        filter_col = np.reshape(self.w,(F,-1))
        print(filter_col.shape)
        conv_it = np.dot(im_col, filter_col.T) + self.bias
        out = col2im(conv_it,H_prime,W_prime)
        return out
    

    
class Pooling:
    def __init__(self,height: int, width: int, stride: int = 2):
        self.height = height
        self.width = width
        self.stride = stride
        
    def max_pooling(self,x):
        self.type = 'max'
        batch, F,  H, W = x.shape
        out_h = (H - self.height) // self.stride + 1
        out_w = (W - self.width) // self.stride + 1
        self.mask  = col = np.zeros((batch, F, out_h , out_w))
        for y in range(out_h):
            for x_pos in range(out_w):
                patch = x[:,:, y*self.stride:y*self.stride+self.height, x_pos*self.stride:x_pos*self.stride+self.width]
                max_patch = patch.max(axis=(2,3))
                col[:, :, y, x_pos] = max_patch
                max_patch[max_patch != 0] = 1
                print(max_patch)
                self.mask[:, :, y, x_pos] = max_patch
        return col
    def average_pooling(self,x):
        self.type = 'average'
        batch, F,  H, W = x.shape
        out_h = (H - self.height) // self.stride + 1
        out_w = (W - self.width) // self.stride + 1
        col = np.zeros((batch, F,  out_h , out_w))
        for y in range(out_h):
            for x_pos in range(out_w):
                patch = x[:,:, y*self.stride:y*self.stride+self.height, x_pos*self.stride:x_pos*self.stride+self.width].mean(axis=(2,3))
                col[:, :, y, x_pos] = patch
        return col
    








            
    