import torch
import torch.nn as nn
import torch.optim as optim
from fc import FullyConnected

def binary_cross_entropy_matrix(recon_x, x):
    # Ensure recon_x values are within (0, 1) to avoid log(0)
    epsilon = 1e-12
    
    # Move tensors to GPU if they are not already there
    if not recon_x.is_cuda:
        recon_x = recon_x.cuda()
    if not x.is_cuda:
        x = x.cuda()
    
    # Clamp the recon_x values to avoid log(0)
    recon_x = torch.clamp(recon_x, epsilon, 1.0 - epsilon)
    
    # Binary cross-entropy calculation
    bce = -torch.sum(x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x))
    
    return bce



class Relu:
    def forward(self, X):
        return torch.maximum(X, torch.tensor(0.0, device=X.device))

    def backward(self, dout, X):
        dx = dout.clone()
        dx[X <= 0] = 0
        return dx

class LeakyRelu:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, X):
        return torch.where(X > 0, X, self.alpha * X)

    def backward(self, dout, X):
        dx = dout.clone()
        dx[X > 0] = 1
        dx[X <= 0] = self.alpha
        return dx

class VAE:
    def __init__(self, input_dim, latent_dim):
        self.encoder_fc1 = FullyConnected(input_dim, 128, device='cuda')
        self.encoder_fc2_mu = FullyConnected(128, latent_dim, device='cuda')
        self.encoder_fc2_logvar = FullyConnected(128, latent_dim, device='cuda')
        
        self.decoder_fc1 = FullyConnected(latent_dim, 128, device='cuda')
        self.decoder_fc2 = FullyConnected(128, input_dim, device='cuda')

        self.relu = Relu()

    def encode(self, x):
        h1 = self.relu.forward(self.encoder_fc1.forward(x))
        mu = self.encoder_fc2_mu.forward(h1)
        logvar = self.encoder_fc2_logvar.forward(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu.forward(self.decoder_fc1.forward(z))
        return torch.sigmoid(self.decoder_fc2.forward(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
 
    
    def update_params(self, lr):
        self.encoder_fc1.update_params(lr)
        self.encoder_fc2_mu.update_params(lr)
        self.encoder_fc2_logvar.update_params(lr)
        self.decoder_fc1.update_params(lr)
        self.decoder_fc2.update_params(lr)




class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = FullyConnected(input_dim, 128)
        self.fc2 = FullyConnected(128, 1)
        self.leaky_relu = LeakyRelu(alpha=0.2)

    def forward(self, x):
        h = self.leaky_relu.forward(self.fc1.forward(x))
        return torch.sigmoid(self.fc2.forward(h))
    def update_params(self, lr):
        self.fc1.update_params(lr)
        self.fc2.update_params(lr)

def vae_loss(recon_x, x, mu, logvar):
    BCE = binary_cross_entropy_matrix(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae_gan(data_loader, vae, discriminator,vae_lr, disc_lr, num_epochs, device):
    for epoch in range(num_epochs):
       
        for batch_idx, (data, labels) in enumerate(data_loader):
          
            data = data.view(data.size(0), -1).to(device)

            recon_data, mu, logvar = vae.forward(data)
            print(recon_data)
            recon_loss = vae_loss(recon_data, data, mu, logvar)
            
            z = vae.reparameterize(mu, logvar)
            fake_data = vae.decode(z)

            
            real_labels = torch.ones(data.size(0), 1, device=device)
            fake_labels = torch.zeros(data.size(0), 1, device=device)

            real_loss = binary_cross_entropy_matrix(discriminator(data), real_labels)
            fake_loss = binary_cross_entropy_matrix(discriminator(fake_data.detach()), fake_labels)
            disc_loss = real_loss + fake_loss

            disc_loss.backward()
            discriminator.update_params(disc_lr)

            # VAE Generator Loss
            gan_loss = binary_cross_entropy_matrix(discriminator(fake_data), real_labels)
            total_loss = recon_loss + gan_loss
            total_loss.backward()
            vae.update_params(vae_lr)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}, D Loss: {disc_loss.item():.4f}")



if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, pin_memory = True)

    # Model parameters
    input_dim = 28 * 28  # MNIST images are 28x28
    latent_dim = 20

    # Initialize models
    vae = VAE(input_dim, latent_dim)
    input_dim = 28 * 28  # MNIST images are 28x28
    latent_dim = 20

    # Initialize models
    vae = VAE(input_dim, latent_dim)
    discriminator = Discriminator(input_dim)

    # Training parameters
    vae_lr = 1e-3
    disc_lr = 1e-3
    num_epochs = 10

    # Train models
    train_vae_gan(train_loader, vae, discriminator, vae_lr, disc_lr, num_epochs, device)



   

   