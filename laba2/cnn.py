import numpy as np
from MLP import MLP
from tqdm import tqdm
from matplotlib import pyplot as plt
rng = np.random.default_rng(51)

def get_indices(X_shape, HF, WF, stride, pad):
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def col2im_B(dX_col, X_shape, HF, WF, stride, pad):
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    
    # Index matrices, necessary to transform our input image into a matrix. 
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]



def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, cache):
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx

def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
    """
    Args:
      dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
      h_prime,w_prime: height and width for the feature map
      strid: stride
      hh,ww,c: size of the filters
    Returns:
      dx: Gradients for x, (C,H,W)
    """
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c,H,W])
    for i in range(h_prime*w_prime):
        
        row = dim_col[i,:]
        h_start = (i / w_prime) * stride
        w_start = (i % w_prime) * stride
       

        dx[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
    return dx


def im2col_(x,hh,ww,stride):
    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

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
        self.t = 1
        self.mW =np.zeros_like(self.w) 
        self.mb = np.zeros_like(self.bias) 
        self.vW =np.zeros_like(self.w) 
        self.vb = np.zeros_like(self.bias) 
        
    def _init_filter_bias(self, amount_filters ):
        return np.zeros((amount_filters,1)) 

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
        self.im_pad = np.pad(x,((0,0),(0,0),(self.pad_num,self.pad_num),(self.pad_num,self.pad_num)),'constant')
        self.im_col = im2col(self.im_pad,HH,WW,self.stride)
        self.filter_col = np.reshape(self.w,(F,-1))
        conv_it = np.dot(self.im_col, self.filter_col.T) + self.bias.T
        out = col2im(conv_it,H_prime,W_prime)
        self.x = x
        return out
    def backward(self, dout, batch_size):
        N,Cx,H,W = self.x.shape
        F, C, HH, WW = self.w.shape
        H_prime = (H+2*self.pad_num-HH) // self.stride + 1
        W_prime = (W+2*self.pad_num-WW) // self.stride + 1
        dout_resh = np.reshape(dout,(F,-1))
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, N))
        dout = np.concatenate(dout, axis=-1)
        self.db = np.sum(dout_resh, axis=1, keepdims=True).reshape(self.bias.shape) 
        self.dw = np.dot(dout, self.im_col.reshape(dout.shape[1],self.im_col.shape[2])).reshape(self.w.shape) 
        dx  = np.dot(self.filter_col.T,dout)
        dx = col2im_B( dx,self.x.shape, self.height, self.width, self.stride, self.pad_num)
       
        return dx
    

    def _update_params(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7):
       
        self.mW  = beta_1*self.mW  + (1-beta_1)*self.dw
        self.mb  = beta_1*self.mb  + (1-beta_1)*self.db 
        # update second moments
        self.vW  = beta_2*self.vW  + (1-beta_2)*(self.dw **2)
        self.vb  = beta_2*self.vb  + (1-beta_2)*(self.db **2)
        # correction
        mW_corr = self.mW  / (1-beta_1**self.t)
        mb_corr = self.mb  / (1-beta_1**self.t)
        vW_corr = self.vW  / (1-beta_2**self.t)
        vb_corr = self.vb  / (1-beta_2**self.t)
        # update params
        self.w  -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
        self.bias  -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        '''print("After update: ", np.sum(self.w), np.sum(self.bias))'''
        # update time
        self.t += 1

    
class Pooling:
    def __init__(self,height: int, width: int, stride: int = 2):
        self.height = height
        self.width = width
        self.stride = stride
    def max_pooling(self,x):
        self.type = 'max'
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
    
    def pool_backward(self, dout):
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

from IPython import display

class CNN:
    def __init__(self):
        self.conv2 = Conv(6, 3,3,1)
        self.pool1 = Pooling(2,2)
        self.conv5 = Conv(3, 4,4,6)
        self.pool2 = Pooling(2,2)
        self.flatten_size = 75
        self.mlp = MLP([self.flatten_size, 25, 10])
         
    def forward(self, X):
        self.conv2_out = self.conv2.conv_forward(X, conv_param={'pad': 0, 'stride': 1})
        self.relu1 = relu(self.conv2_out)
        self.pool1_out = self.pool1.max_pooling(self.relu1)
        self.conv5_out = self.conv5.conv_forward(self.pool1_out, conv_param={'pad': 0, 'stride': 1})
        self.relu2 = relu(self.conv5_out)
        self.pool2_out = self.pool2.max_pooling(self.relu2)
        self.flatten = self.pool2_out.reshape(self.pool2_out.shape[0], -1)
        y_pred = self.mlp._feedforward(self.flatten)
        return y_pred

    def train(self, X, y, epochs=1, batch_size=8, lr=1e-2):
        X = np.array(X) 
        y = np.array(y)
        n_samples = y.shape[0]
        epoch_losses = []
        train_costs = []
        plt.ion()  # Включение интерактивного режима
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Средняя потеря')
        ax.set_title('Обучение модели')
        ax.set_xlim(1, epochs)
        ax.set_ylim(0, 1)  # Предварительная установка, будет автоматически изменяться
        line, = ax.plot([], [], 'b-', label='Loss (train)')
        ax.legend(loc='upper right')
        display.display(fig)

        for epoch in range(epochs):
            ax.cla()
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            train_loss = 0
            train_acc = 0 
            pbar = tqdm(range(0, n_samples, batch_size))
            for start_idx in pbar:
                
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                y_pred = self.forward(X_batch)
                dout = self.backward(y_batch,end_idx-start_idx)
                loss = self.mlp._compute_loss(y_pred, y_batch) 
                train_loss += loss 
                train_acc += sum((np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))
                pbar.set_postfix({'acc': train_acc, 'train_loss': train_loss})
           
            train_loss /= n_samples
            train_costs.append(train_loss)
            train_acc /= n_samples            
            print(f"Эпоха {epoch+1}/{epochs}, Средняя потеря: {train_loss}")
            # Обновление графика после каждой эпохи
            line.set_xdata(range(1, epoch + 2))
            line.set_ydata(train_costs)
            
            # Автоматическое обновление пределов оси Y
            ax.set_xlim(1, max(epochs, epoch + 1))
            ax.set_ylim(0, max(train_costs) * 1.1)
            
            # Обновление графика в ячейке
            display.clear_output(wait=True)
            display.display(fig)
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        return train_costs
    
    def backward(self, y_batch,batch_size):
        self.mlp._backprop(y_batch)
        self.mlp._update_params_Adam()
        dout = np.dot(self.mlp.delta,self.mlp.W[0]).reshape(batch_size, 3,5,5)
        dout = self.pool2.pool_backward(dout)
        dout = relu_backward(dout, self.conv5_out)
        dout = self.conv5.backward(dout, batch_size)
        self.conv5._update_params()
        dout = self.pool1.pool_backward(dout)
        dout = relu_backward(dout, self.conv2_out)
        dout = self.conv2.backward(dout, batch_size)
        self.conv2._update_params()
        return dout
    def predict(self, X):
        self.conv2_out = self.conv2.conv_forward(X, conv_param={'pad': 0, 'stride': 1})
        self.relu1 = relu(self.conv2_out)
        self.pool1_out = self.pool1.max_pooling(self.relu1)
        self.conv5_out = self.conv5.conv_forward(self.pool1_out, conv_param={'pad': 0, 'stride': 1})
        self.relu2 = relu(self.conv5_out)
        self.pool2_out = self.pool2.max_pooling(self.relu2)
        self.flatten = self.pool2_out.reshape(X.shape[0], -1)
        self.flatten_size = self.flatten.shape[1]
        predict = self.mlp.predict(self.flatten)
        return np.argmax(predict, axis=1)
    
