import torch
import torch.nn.functional as F
from activations import Sigmoid_t, Relu_torch, LeakyRelu_torch





class FullyConnected:
    def __init__(self, input_size, output_size, device=None):
        self.device = device if device else torch.device('cpu')
        self.W = self._init_weights(input_size, output_size).to(self.device)
        self.b = torch.zeros(output_size, device=self.device)

    def forward(self, X):
        self.input = X.to(self.device)
        return torch.matmul(self.input, self.W.T) + self.b

    def backward(self, grad_output):
        grad_output = grad_output.to(self.device)
        self.dW = torch.matmul(grad_output.T, self.input) / self.input.shape[0]
        self.db = torch.sum(grad_output, dim=0) / self.input.shape[0]
        return torch.matmul(grad_output, self.W)

    def update_params(self, lr=0.001):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def _init_weights(self, input_size, output_size):
        limit = (6.0 / (input_size + output_size)) ** 0.5
        return torch.empty(output_size, input_size).uniform_(-limit, limit)

class Encoder:
    def __init__(self, input_dim, hidden_dims, latent_dim, device=None):
        self.device = device if device else torch.device('cpu')
        self.layers = [
            FullyConnected(input_dim, hidden_dims[0], device),
            FullyConnected(hidden_dims[0], hidden_dims[1], device),
        ]
        self.mu_layer = FullyConnected(hidden_dims[1], latent_dim, device)
        self.logvar_layer = FullyConnected(hidden_dims[1], latent_dim, device)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        mu = self.mu_layer.forward(h)
        logvar = self.logvar_layer.forward(h)
        reconst = self.reparameterize(mu, logvar)
        return reconst, mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def backward(self, dmu, dlogvar):
        grad = self.mu_layer.backward(dmu) + self.logvar_layer.backward(dlogvar)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_params(self, lr):
        for layer in self.layers + [self.mu_layer, self.logvar_layer]:
            layer.update_params(lr)

class Decoder:
    def __init__(self, latent_dim, hidden_dims, output_dim, device=None):
        self.device = device if device else torch.device('cpu')
        self.layers = [
            FullyConnected(latent_dim, hidden_dims[1], device),
            FullyConnected(hidden_dims[1], hidden_dims[0], device),
            FullyConnected(hidden_dims[0], output_dim, device),
            Sigmoid_t()
        ]
    def forward(self, z):
        h = z
        for layer in self.layers[:-1]:
            h = layer.forward(h)
        return h

    def backward(self, dout):
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_params(self, lr):
        for layer in self.layers:
            if 'activations' not in str(layer.__class__):
                layer.update_params(lr)

class Discriminator:
    def __init__(self, input_dim, hidden_dims, device=None):
        self.device = device if device else torch.device('cpu')
        self.layers = [
            FullyConnected(input_dim, hidden_dims[0], device),
            FullyConnected(hidden_dims[0], hidden_dims[1], device),
            FullyConnected(hidden_dims[1], 1, device),
        ]

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = F.relu(layer.forward(h))
        return torch.sigmoid(self.layers[-1].forward(h))

    def backward(self, dout):
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_params(self, lr):
        for layer in self.layers:
            layer.update_params(lr)





# Load MNIST dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Example usage:

class VAE:
    def __init__(self,encoder,decoder,device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def compute_vae_loss(self, x, x_reconstructed, mu, logvar):
        reconstruction_loss = torch.mean(torch.sum((x - x_reconstructed) ** 2, dim=1))
        kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))
        return reconstruction_loss, kl_divergence

    def forward(self, X):
        z, mu, logvar = encoder.forward(X)
        reconstructed_data  = decoder.forward(z)
        return reconstructed_data, mu, logvar
    def backward(self, x, x_reconstructed, mu, logvar):
            # Backpropagation for VAE
            d_reconstruction = 2 * (x_reconstructed - x) / 64
            d_decoder = self.decoder.backward(d_reconstruction)
            d_mu = -mu / x.shape[0]
            d_logvar = (-0.5 * (1 - torch.exp(logvar)) ) / x.shape[0]
            self.encoder.backward(d_mu, d_logvar)
    def _update_params(self, lr = 1e-4):
        self.decoder.update_params(lr)
        self.encoder.update_params(lr)



'''
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder(784, [256, 128], 50, device=device)
decoder = Decoder(50, [256, 128], 784, device=device)
discriminator = Discriminator(784, [256, 128], device=device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, pin_memory = True)

vae = VAE(encoder, decoder,device='cuda')

for batch_idx, (data, labels) in enumerate(train_loader):
    data = data.view(data.size(0), -1).to(device)
    reconstructed_data, mu, logvar = vae.forward(data)
    reconstruction_loss, kl_divergence = vae.compute_vae_loss(data,reconstructed_data , mu,logvar)
    vae.backward(x = data, x_reconstructed=reconstructed_data,mu = mu, logvar=logvar )
    print(
        'ITER'

    )'''
    
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация модели
encoder = Encoder(784, [256, 128], 50, device=device)
decoder = Decoder(50, [256, 128], 784, device=device)
vae = VAE(encoder, decoder, device=device)

# Датасет и DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, pin_memory=True)


# Количество эпох
num_epochs = 10

# Цикл обучения
for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    epoch_recon_loss = 0
    epoch_kl_div = 0
    num_batches = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        reconstructed_data, mu, logvar = vae.forward(data)
        reconstruction_loss, kl_divergence = vae.compute_vae_loss(data,reconstructed_data , mu,logvar)
        vae.backward(x = data, x_reconstructed=reconstructed_data,mu = mu, logvar=logvar )
        vae._update_params(lr = 1e-5)

        # Суммирование метрик для текущей эпохи
        epoch_recon_loss += reconstruction_loss.item()
        epoch_kl_div += kl_divergence.item()
        num_batches += 1

  
    avg_recon_loss = epoch_recon_loss / num_batches
    avg_kl_div = epoch_kl_div / num_batches

   
    epoch_time = time.time() - start_time

  
    print(
        f"Epoch {epoch}/{num_epochs} | "
        f"Reconstruction Loss: {avg_recon_loss:.4f} | "
        f"KL Divergence: {avg_kl_div:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )
