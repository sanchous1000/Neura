import cupy as cp

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
    def __init__(self, input_size, output_size):
        limit = cp.sqrt(6. / (input_size + output_size))
        self.W = cp.random.uniform(-limit, limit, size=(output_size, input_size))
        self.b = cp.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        return cp.dot(X, self.W.T) + self.b

    def backward(self, dout):
        self.dW = cp.dot(dout.T, self.X)
        self.db = cp.sum(dout, axis=0, keepdims=True)
        return cp.dot(dout, self.W)


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


class MultiHeadAttention:
    def __init__(self, n_heads, head_size, n_emb):
        self.heads = [Head(head_size, n_emb) for _ in range(n_heads)]
        self.proj = FullyConnected(n_heads * head_size, n_emb)

    def forward(self, X):
        head_outputs = [head.forward(X) for head in self.heads]
        concat = cp.concatenate(head_outputs, axis=-1)
        return self.proj.forward(concat)


class Feedforward:
    def __init__(self, n_emb):
        self.fc1 = FullyConnected(n_emb, 4 * n_emb)
        self.relu = Relu()
        self.fc2 = FullyConnected(4 * n_emb, n_emb)

    def forward(self, X):
        X = self.fc1.forward(X)
        X = self.relu.forward(X)
        return self.fc2.forward(X)


class BlockEncoder:
    def __init__(self, n_emb, n_heads):
        head_size = n_emb // n_heads
        self.mha = MultiHeadAttention(n_heads, head_size, n_emb)
        self.ln1 = LayerNorm(n_emb)
        self.ff = Feedforward(n_emb)
        self.ln2 = LayerNorm(n_emb)

    def forward(self, X):
        attn_output = self.mha.forward(X)
        X = X + attn_output
        X = self.ln1.forward(X)
        ff_output = self.ff.forward(X)
        X = X + ff_output
        return self.ln2.forward(X)


class Classification:
    def __init__(self, n_emb, n_classes):
        self.fc = FullyConnected(n_emb, n_classes)
        self.softmax = Softmax()

    def forward(self, X):
        X = self.fc.forward(X)
        return self.softmax.forward(X)


# Sample usage:
n_emb = 64
n_heads = 8
n_classes = 2
batch_size = 16
seq_len = 10

X = cp.random.randn(batch_size, seq_len, n_emb)  # Example input
encoder = BlockEncoder(n_emb, n_heads)
classifier = Classification(n_emb, n_classes)

# Forward pass
encoded = encoder.forward(X)
predictions = classifier.forward(encoded)
