import cupy as cp
import torch 
import numpy as np
class Relu:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return cp.maximum(X, 0)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.X <= 0] = 0
        return dx


class LayerNorm:
    def __init__(self, input_dim, epsilon=1e-5):
        self.input_dim = input_dim
        self.gamma = cp.ones((1, input_dim))
        self.beta = cp.zeros((1, input_dim))
        self.epsilon = epsilon

    def forward(self, x):
        self.x = x
        self.mean = cp.mean(x, axis=1, keepdims=True)
        self.variance = cp.var(x, axis=1, keepdims=True)
        self.x_norm = (x - self.mean) / cp.sqrt(self.variance + self.epsilon)
        self.out = self.gamma * self.x_norm + self.beta
        return self.out

    def backward(self, dout):
        self.dW = cp.sum(dout * self.x_norm, axis=(1, 2))
        self.db = cp.sum(dout, axis=(1, 2))

        dx_norm = dout * self.gamma
        dvar = cp.sum(dx_norm * (self.x - self.mean) * -0.5 * cp.power(self.variance + self.epsilon, -1.5), axis=1, keepdims=True)
        dmean = cp.sum(dx_norm * -1 / cp.sqrt(self.variance + self.epsilon), axis=1, keepdims=True) + \
                dvar * cp.sum(-2 * (self.x - self.mean), axis=1, keepdims=True) / self.input_dim
        dx = dx_norm / cp.sqrt(self.variance + self.epsilon) + \
             dvar * 2 * (self.x - self.mean) / self.input_dim + \
             dmean / self.input_dim
        return dx


class FullyConnected:
    def __init__(self, input_size, output_size, type ='transform' ):
        limit = cp.sqrt(6. / (input_size + output_size))
        self.W = cp.random.uniform(-limit, limit, size=(output_size, input_size))
        self.b = cp.zeros((1, output_size))
        self.type = type

    def forward(self, X):
        self.a_l = X
        self.X = X
        return cp.dot(X, self.W.T) + self.b

    def backward(self, dout):
        m =  self.a_l.shape[0]
        if self.type == 'transform':
            self.dW =  cp.sum(np.matmul(dout.transpose(0, 2, 1), self.a_l), axis = 0)
            self.db = cp.sum(dout, axis = (0, 1))
        else:
            self.dW = cp.dot(dout.T, self.a_l) / m 

            self.db = cp.sum(dout, axis = 0, keepdims=True) / m
        delta = cp.dot(dout, self.W)
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
        self.W -= lr*mW_corr / (cp.sqrt(vW_corr)+eps)
        self.b  -= lr*mb_corr / (cp.sqrt(vb_corr)+eps)
        self.t += 1


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, X):
        exp_X = cp.exp(X - cp.max(X, axis=-1, keepdims=True))
        self.output = exp_X / cp.sum(exp_X, axis=-1, keepdims=True)
        return self.output

    def backward(self, dout):
        return self.output * (dout - cp.sum(dout * self.output, axis=-1, keepdims=True))


class Head:
    def __init__(self, head_size, C):
        self.k = FullyConnected(C, head_size)
        self.q = FullyConnected(C, head_size)
        self.v = FullyConnected(C, head_size)
        self.softmax = Softmax()
        self.head_size = head_size

    def forward(self, X):
        key = self.k.forward(X)
        query = self.q.forward(X)
        value = self.v.forward(X)
        scores = cp.matmul(query, key.transpose(0, 2, 1))
        scores = scores / cp.sqrt(self.head_size)
        attn_weights = self.softmax.forward(scores)
        return cp.matmul(attn_weights, value)
    
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
    def __init__(self, n_heads, head_size, n_emb):
        self.heads = [Head(head_size, n_emb) for _ in range(n_heads)]
        self.proj = FullyConnected(n_heads * head_size, n_emb)

    def forward(self, X):
        head_outputs = [head.forward(X) for head in self.heads]
        concat = cp.concatenate(head_outputs, axis=-1)
        return self.proj.forward(concat)
    def backward(self, dOut):
        # Backward через проекционный слой
        dConcat = self.proj.backward(dOut)
        # Backward через все головы
        dX_heads = []
        split_dOut = cp.split(dConcat, len(self.heads), axis=-1)
        for i, head in enumerate(self.heads):
            dX_heads.append(head.backward(split_dOut[i]))
        dX = cp.sum(cp.stack(dX_heads), axis=0)
        return dX
    def update_params(self, lr = 1e-6):
        self.proj.update_params(lr) 


class Feedforward:
    def __init__(self,n_emb ):
        self.fc1 = FullyConnected(n_emb, 4 * n_emb)
        self.relu = Relu()
        self.fc2 = FullyConnected(4 * n_emb, n_emb)
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


class BlockEncoder:
    def __init__(self, n_emb, n_heads):
        head_size = n_emb//n_heads
        self.self_att = MultiHeadAttention(n_heads, head_size, n_emb)
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

    
class Meaner:
    def __init__(self):
        pass 
    def forward(self, X):
        self.time = X.shape[1]
        return cp.mean(X, axis = 1)
    def backward(self, dout):
        din = dout[:, np.newaxis, :] / self.time  # Расширяем размерность вдоль оси tim
        return cp.repeat(din, self.time, axis=1) 





class Classification:
    def __init__(self,input,output ):
        self.meaner = Meaner()
        self.fc1 = FullyConnected(input, output, type = '34')
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





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel






class SentimentEncoder:
    def __init__(self, n_emb, n_heads, output_classes):
        self.encoder = BlockEncoder(n_emb=n_emb, n_heads=n_heads)
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


tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model_pre = AutoModel.from_pretrained("cointegrated/rubert-tiny")

def binary_cross_entropy_loss(y_true, y_pred):
    """
    Кросс-энтропийный лосс для бинарной классификации.

    Args:
        y_true: Истинные метки (batch_size, ) или (batch_size, 1).
        y_pred: Предсказанные вероятности для положительного класса (batch_size, ).

    Returns:
        Средний лосс по батчу.
    """
    # Ensure y_true is flattened
    y_true = y_true.flatten()  # Shape: (batch_size, )
    
    # Compute binary cross-entropy loss
    loss = -cp.mean(
        y_true * cp.log(y_pred + 1e-8) + (1 - y_true) * cp.log(1 - y_pred + 1e-8)
    )
    return loss



data = pd.read_csv(r'C:\Users\Aleks\Documents\Neura\lab5\russian_comments_from_2ch_pikabu.csv')
data['tok_len'] = [len(tokenizer(i)['input_ids']) for i in data.comment]
data = data[data.tok_len < 450]

mas = np.zeros((len(data),2))
for i, v in enumerate(data.toxic):
    mas[i,int(v)] = 1

X_train,X_test, y_train, y_test = train_test_split(data['comment'], mas,  train_size = 0.8)
from tqdm import tqdm
import cupy as cp
from transformers import AutoTokenizer, AutoModel

# Инициализация модели
n_emb = 30  # размер эмбеддингов
n_heads = 2  # число голов
output_classes = 2  # бинарная классификация
block_size = 512
batch_size = 128
epochs = 3

model = SentimentEncoder(n_emb=n_emb, n_heads=n_heads, output_classes=output_classes)

# Предобученная модель и токенизатор
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model_pre = AutoModel.from_pretrained("cointegrated/rubert-tiny")

# Токенизация данных перед обучением
X_train = list(X_train)
y_train_gpu = cp.array(y_train, dtype=cp.float32)  # Метки переводим на GPU

for epoch in range(epochs):
    total_loss = 0
    indices = cp.arange(len(X_train))
    cp.random.shuffle(indices)

    for batch_start in tqdm(range(0, len(X_train), batch_size)):
        batch_end = batch_start + batch_size
        batch_indices = indices[batch_start:batch_end].get()  # Преобразуем индексы для использования с PyTorch
        
        # Формируем батч
        X_batch = [X_train[idx] for idx in batch_indices]
        y_batch = y_train_gpu[batch_indices]

        # Токенизация текста
        inputs = tokenizer(X_batch, padding=True, truncation=True, max_length=block_size, return_tensors="pt")
        
        # Forward pass через предобученную модель
        outputs = model_pre(**inputs)
        text_embeddings = cp.array(outputs.last_hidden_state[:, :, :n_emb].detach().cpu().numpy())  # GPU embeddings
        
        # Forward pass через кастомную модель
        logits = cp.array(model.forward(text_embeddings))
        ''' loss = binary_cross_entropy_loss(y_batch, logits)
        total_loss += loss.get()  # Суммируем значения лосса на CPU'''

        # Backward pass
        model.backward(logits, y_batch)
        model.update_params(lr=1e-5)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
