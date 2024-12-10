import numpy as np
from main_cl import Fullyconnected, Flatten, Conv
from activations import LeakyRelu, Tanh, Sigmoid

import numpy as np

import numpy as np



class Encoder:
    def __init__(self, arch = (784, 256, 128), zdims=50, learning_rate  = 1e-8):
        self.zdims = zdims
        self.fc1 = Fullyconnected(input_size=arch[0], output_size=arch[1])
        self.LeakyRelu1 =  LeakyRelu()
        self.fc2 = Fullyconnected(input_size=arch[1], output_size=arch[2])
        self.LeakyRelu2 =  LeakyRelu()
        self.mu = Fullyconnected(input_size=arch[2], output_size=zdims)
        self.log_var = Fullyconnected(input_size=arch[2], output_size=zdims)
        self.lr = learning_rate


        self.layers = [
                       self.fc1,
                     self.LeakyRelu1,
                     self.fc2,
                     self.LeakyRelu2,
                     [self.mu,
                      self.log_var]]

        

    def forward(self, x):
        # Прямой проход
        h = x
        for layer in self.layers:
            if type(layer) != list:
                h = layer.forward(h)
            else:
                mu = layer[0].forward(h)
                logvar = layer[1].forward(h)
        return mu, logvar

    def backward(self, dmu, dlogvar):
        # Градиенты для mu и logvar
   
        for layer in reversed(self.layers):
            if type(layer) == list:
                dh = layer[0].backward(dmu) + layer[1].backward(dlogvar)
            else:
                if 'Flatten' in str(layer.__class__):
                    continue
                dh = layer.backward(dh)
        return dh


    def reparameterize(self, mu, logvar):
        std = np.exp(logvar)
        eps = np.random.randn(*logvar.shape)
        return mu + eps * std
    
    def update_params(self):
        for layer in self.layers:
            if 'activations' not in str(layer.__class__):
                if 'Flatten' not in str(layer.__class__):
                    if type(layer) == list:
                        layer[0].update_params(self.lr)
                        layer[1].update_params(self.lr)
                    else:
                        layer.update_params(self.lr)
            

class Decoder:
    def __init__(self, zdims=50, arch = (784, 256, 128), learning_rate  = 1e-8):
        self.de_x_1 = Fullyconnected(input_size=zdims, output_size=arch[2])
        self.LeakyRelu1 = LeakyRelu()
        self.de_x_2 = Fullyconnected(input_size=arch[2], output_size=arch[1])
        self.LeakyRelu2 = LeakyRelu()
        self.de_x_3 = Fullyconnected(input_size=arch[1], output_size=arch[0])
        self.sig = Sigmoid()
        self.lr = learning_rate
        self.layers = [self.de_x_1,
                        self.LeakyRelu1,
                        self.de_x_2,
                        self.LeakyRelu2,
                        self.de_x_3,
                        self.sig]


    def forward(self, z):
        for layer in self.layers:
            z = layer.forward(z)
        return z

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def update_params(self):
        for layer in self.layers:
             if 'activations' not in str(layer.__class__):
                if 'Flatten' not in str(layer.__class__):
                    layer.update_params(self.lr)
        

class Discriminator:
    def __init__(self, input_size=784, arch=(256, 128), learning_rate=1e-8):
        self.fc1 = Fullyconnected(input_size=input_size, output_size=arch[0])
        self.LeakyRelu1 = LeakyRelu()
        self.fc2 = Fullyconnected(input_size=arch[0], output_size=arch[1])
        self.LeakyRelu2 = LeakyRelu()
        self.fc3 = Fullyconnected(input_size=arch[1], output_size=50)
        self.sig = Sigmoid()
        self.lr = learning_rate

        self.layers = [
            self.fc1,
            self.LeakyRelu1,
            self.fc2,
            self.LeakyRelu2,
            self.fc3,
            self.sig
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def update_params(self):
        for layer in self.layers:
            if 'activations' not in str(layer.__class__):
                if 'Flatten' not in str(layer.__class__):
                    layer.update_params(self.lr)


        


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class VAEGANTrainer:
    def __init__(self, encoder, decoder, discriminator):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        mu, logvar = self.encoder.forward(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_reconstructed = self.decoder.forward(z)
        return x_reconstructed, mu, logvar, z

    def backward(self, x, x_reconstructed, mu, logvar):
        # Backpropagation for VAE
        d_reconstruction = 2 * (x_reconstructed - x) / x.shape[0]
        d_decoder = self.decoder.backward(d_reconstruction)
        d_mu = -mu / x.shape[0]
        d_logvar = -0.5 * (1 - np.exp(logvar)) / x.shape[0]
        self.encoder.backward(d_mu, d_logvar)

    def compute_vae_loss(self, x, x_reconstructed, mu, logvar):
        # VAE Loss
        reconstruction_loss = np.sum((x - x_reconstructed)**2) / x.shape[0]
        kl_divergence = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / x.shape[0]
        return reconstruction_loss, kl_divergence

    def compute_gan_loss(self, x, x_reconstructed):
        # GAN Loss
        real_labels = np.ones((x.shape[0], 1))  # Real labels
        fake_labels = np.zeros((x.shape[0], 1))  # Fake labels

        real_preds = self.discriminator.forward(x)
        fake_preds = self.discriminator.forward(x_reconstructed)

        d_loss_real = -np.mean(np.log(real_preds + 1e-8))
        d_loss_fake = -np.mean(np.log(1 - fake_preds + 1e-8))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -np.mean(np.log(fake_preds + 1e-8))
        return d_loss, g_loss, real_preds, fake_preds

    def train(self, x_train, batch_size=128, epochs=50):
        self.vae_loss_history, self.gan_loss_history, self.recon_loss_history, self.kl_loss_history = [], [], [], []

        for epoch in range(epochs):
            total_vae_loss, total_gan_loss = 0, 0
            recon_loss_, kl_loss = 0, 0

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                batch_size_actual = x_batch.shape[0]

                # VAE Forward
                x_reconstructed, mu, logvar, z = self.forward(x_batch)

                # Compute losses
                recon_loss, kl = self.compute_vae_loss(x_batch, x_reconstructed, mu, logvar)
                total_vae_loss += recon_loss + kl
                recon_loss_ += recon_loss
                kl_loss += kl

                # VAE Backward
                self.backward(x_batch, x_reconstructed, mu, logvar)

                # Update Encoder and Decoder
                for model in [self.encoder, self.decoder]:
                    model.update_params()

                # GAN Forward and Loss
                # GAN Forward and Loss
                d_loss, g_loss, real_preds, fake_preds = self.compute_gan_loss(x_batch, x_reconstructed)
                total_gan_loss += d_loss

                # Backward for Discriminator
                d_real = -1 / (real_preds + 1e-8)
                d_fake = 1 / (1 - fake_preds + 1e-8)
                self.discriminator.backward(d_real + d_fake)
                self.discriminator.update_params()

                # Backward for Generator (Decoder)
                d_fake_gen = np.ones_like(x_reconstructed) * -1 / (fake_preds + 1e-8)
                self.decoder.backward(d_fake_gen)
                self.decoder.update_params()


            # Logging
            avg_vae_loss = total_vae_loss / (len(x_train) / batch_size)
            avg_gan_loss = total_gan_loss / (len(x_train) / batch_size)
            self.vae_loss_history.append(avg_vae_loss)
            self.gan_loss_history.append(avg_gan_loss)
            self.recon_loss_history.append(recon_loss_ / (len(x_train) / batch_size))
            self.kl_loss_history.append(kl_loss / (len(x_train) / batch_size))

            print(f"Epoch {epoch + 1}/{epochs} - VAE Loss: {avg_vae_loss:.4f} - GAN Loss: {avg_gan_loss:.4f} - Recon Loss: {recon_loss_ / (len(x_train) / batch_size):.4f} - KL Loss: {kl_loss / (len(x_train) / batch_size):.4f}")

        '''# Plot metrics
        self.plot_metrics(vae_loss_history, gan_loss_history, kl_loss_history)
        return vae_loss_history, gan_loss_history, recon_loss_history, kl_loss_history
'''
    @staticmethod
    def plot_metrics(vae_loss_history, gan_loss_history, kl_loss_history):
        plt.figure(figsize=(12, 6))
        plt.plot(vae_loss_history, label="VAE Loss")
        plt.plot(gan_loss_history, label="GAN Loss")
        plt.plot(kl_loss_history, label="KL Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.show()




'''
class VAEGANTrainer:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        #self.discriminator = discriminator
        #self.learning_rate = learning_rate
    def forward(self, x):
        mu, logvar = self.encoder.forward(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_reconstructed = self.decoder.forward(z)
        return x_reconstructed, mu, logvar
    def backward(self, x, x_reconstructed, mu, logvar):
        d_reconstruction = 2 * (x_reconstructed - x) / x.shape[0]
        d_decoder = self.decoder.backward(d_reconstruction)
        d_mu = -mu / x.shape[0]
        d_logvar = -0.5 * (1 - np.exp(logvar)) / x.shape[0]
        self.encoder.backward(d_mu, d_logvar)
    def compute_loss(self, x, x_reconstructed, mu, logvar):
        reconstruction_loss = np.sum((x - x_reconstructed)**2) / x.shape[0]
        kl_divergence = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / x.shape[0]
        return reconstruction_loss,  kl_divergence, 

    def train(self, x_train, batch_size=128, epochs=50):
        vae_loss_history = []
        recon_loss_history, kl_loss_history = [], []
        for epoch in range(epochs):
            total_vae_loss = 0
            recon_loss_ = 0
            kl_loss = 0
            

            for i in range(0, len(x_train), batch_size):
                # Формируем батч
                x_batch = x_train[i:i + batch_size]
                batch_size_actual = x_batch.shape[0]

                x_reconstructed, mu, logvar = self.forward(x_batch)
                
                recon_loss, kl = self.compute_loss(x_batch, x_reconstructed, mu, logvar)
                
                total_vae_loss +=  recon_loss +  kl
                kl_loss += kl
                recon_loss_ += recon_loss
                self.backward(x_batch, x_reconstructed, mu, logvar)

                for model in [self.encoder, self.decoder]:
                    model.update_params()

            avg_loss = total_vae_loss / (len(x_train) / batch_size)
            vae_loss_history.append(avg_loss)
            recon_loss_history.append(recon_loss_ / (len(x_train) / batch_size) )
            kl_loss_history.append(kl_loss/ (len(x_train) / batch_size)  )

            # Вывод метрик
            print(f"Epoch {epoch + 1}/{epochs} - VAE Loss: {avg_loss:.4f} - recon_loss: {recon_loss_ / (len(x_train) / batch_size):.4f} - KL Loss: {kl_loss/ (len(x_train) / batch_size):.4f} -")

        # Построение графиков
        self.plot_metrics(vae_loss_history, recon_loss_history, kl_loss_history)
        return vae_loss_history, recon_loss_history,kl_loss_history

    @staticmethod
    def plot_metrics(vae_loss_history, gan_loss_history, kl_loss_history):
        plt.figure(figsize=(12, 6))
        plt.plot(vae_loss_history, label="VAE Loss")
        plt.plot(gan_loss_history, label="GAN Loss")
        plt.plot(kl_loss_history, label="KL Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.show()

'''