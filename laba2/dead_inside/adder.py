import numpy as np
def get_indices(X_shape, HF, WF, stride, pad):
    m, n_C, n_H, n_W = X_shape
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
    level1 = np.repeat(np.arange(HF), WF)
    level1 = np.tile(level1, n_C)
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    slide1 = np.tile(np.arange(WF), HF)
    slide1 = np.tile(slide1, n_C)
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def col2im_B(dX_col, X_shape, HF, WF, stride, pad):
    N, D, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]

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




'''

def im2col(X, HF, WF, stride, pad):
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    N, D, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
class Conv:
    def __init__(self,amount_filters: int, height: int, width: int, chanels_amount: int = 1, padding: int = 0, stride:int = 1):        
        self.w = self._init_filter_weights((chanels_amount, height, width), amount_filters)
        self.bias = self._init_filter_bias(amount_filters)
        self.amount_filters = amount_filters
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

    def _init_filter_bias(self, amount_filters):
        return np.zeros((1,amount_filters)) 

    def _init_filter_weights(self, arch, amount_filters):
        channels_amount = arch[0]
        height = arch[1]
        width = arch[2]
        limit = np.sqrt(6. / (channels_amount * height * width + channels_amount * height * width))
        return rng.uniform(-limit, limit + 1e-5, size=(amount_filters, channels_amount, height, width))
    
    def forward(self, X):
        
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        self.N_c = n_C_prev

        n_C =  self.amount_filters
        n_H = int((n_H_prev + 2 * self.padding - self.height)/ self.stride) + 1
        n_W = int((n_W_prev + 2 * self.padding - self.width)/ self.stride) + 1
        
        X_col = im2col(X, self.height, self.width, self.stride, self.padding)
        w_col = self.w.reshape((self.amount_filters, -1))
        b_col = self.bias.reshape(-1, 1)
        
        
        out = w_col @ X_col + b_col
      
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.cache = X, X_col, w_col
        return out
    
    def backward(self, dout):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.

            Parameters:
            - dout: error from previous layer.
            
            Returns:
            - dX: error of the current convolutional layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape
        # Compute bias gradient.
        self.db = np.sum(dout, axis=(0,2,3))
        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dout
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dW = dout @ X_col.T
        # Reshape back to image (col2im).
        dX = col2im(dX_col, X.shape, self.height, self.width, self.stride, self.padding)
        # Reshape dw_col into dw.
        self.dW = dW.reshape((dW.shape[0],  self.N_c, self.height, self.width))
                
        return dX
    
    def _update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08): 
        self.mW  = beta_1*self.mW  + (1-beta_1)*self.dW
        self.mb  = beta_1*self.mb  + (1-beta_1)*self.db 
        self.vW  = beta_2*self.vW  + (1-beta_2)*(self.dW **2)
        self.vb  = beta_2*self.vb  + (1-beta_2)*(self.db **2)
        mW_corr = self.mW  / (1-beta_1**self.t)
        mb_corr = self.mb  / (1-beta_1**self.t)
        vW_corr = self.vW  / (1-beta_2**self.t)
        vb_corr = self.vb  / (1-beta_2**self.t)
        self.w  -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
        self.bias  -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        self.t += 1
'''