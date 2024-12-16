import numpy as np
from main_cl import Fullyconnected
from activations import LeakyRelu, Tanh, Sigmoid, Relu

import numpy as np

    


class Encoder:
    def __init__(self, arch = (784, 256), zdims=50, learning_rate  = 1e-8):
        self.zdims = zdims
        self.fc1 = Fullyconnected(input_size=arch[0], output_size=arch[1])
        self.LeakyRelu1 =  Relu()
        #self.fc2 = Fullyconnected(input_size=arch[1], output_size=arch[2])
        #self.LeakyRelu2 =  LeakyRelu()
        self.mu = Fullyconnected(input_size=arch[1], output_size=zdims)
        self.log_var = Fullyconnected(input_size=arch[1], output_size=zdims)
        self.lr = learning_rate


        self.layers = [
                       self.fc1,
                     self.LeakyRelu1,
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
        return mu + eps * std, eps
    
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
    def __init__(self, zdims=50, arch = (784, 256), learning_rate  = 1e-8):
        self.de_x_1 = Fullyconnected(input_size=zdims, output_size=arch[1])
        ''' self.LeakyRelu1 = LeakyRelu()
        self.de_x_2 = Fullyconnected(input_size=arch[2], output_size=arch[1])
        self.LeakyRelu2 = LeakyRelu()'''
        self.de_x_3 = Fullyconnected(input_size=arch[1], output_size=arch[0])
       # self.sig = Sigmoid()
        self.lr = learning_rate
        self.zdims=zdims
        self.layers = [self.de_x_1,
                       # self.LeakyRelu1,
                       # self.de_x_2,
                        #self.LeakyRelu2,
                        self.de_x_3]


    def forward(self, z):
        for layer in self.layers:
            z = layer.forward(z)
        return z

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def update_params(self, lr = None):
        if lr == None:
            lr = self.lr
        for layer in self.layers:
             if 'activations' not in str(layer.__class__):
                if 'Flatten' not in str(layer.__class__):
                    layer.update_params(lr)
        

class Discriminator:
    def __init__(self, input_size=784, arch=(256, 128, 64), learning_rate=1e-8):
        self.fc1 = Fullyconnected(input_size=input_size, output_size=arch[0])
        self.LeakyRelu1 = Relu()
        self.fc2 = Fullyconnected(input_size=arch[0], output_size=arch[1])
        self.LeakyRelu2 = Relu()
        self.fc3 = Fullyconnected(input_size=arch[1], output_size=arch[1])
        self.LeakyRelu3 = Relu()
        self.fc4 = Fullyconnected(input_size=arch[1], output_size=1)
        self.sig = Sigmoid()
        self.lr = learning_rate

        self.layers = [
            self.fc1,
            self.LeakyRelu1,
            self.fc2,
            self.LeakyRelu2,
            self.fc3,
            self.LeakyRelu3,
            self.fc4,
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
    def backward_(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward_(dout)
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
    def __init__(self, encoder, decoder, beta = 1):
        self.encoder = encoder
        self.decoder = decoder

        self.beta = beta

    def forward(self, x):
        mu, logvar = self.encoder.forward(x)
        z , self.eps = self.encoder.reparameterize(mu, logvar)
        x_reconstructed = self.decoder.forward(z)
        return x_reconstructed, mu, logvar, z

    def backward(self, x, x_reconstructed, mu, logvar):
        # Backpropagation for VAE
        d_reconstruction = 2 * (x_reconstructed - x) / x.shape[0]
        d_decoder = self.decoder.backward(d_reconstruction)
        d_mu = (d_decoder + self.beta *  mu) / x.shape[0]
        d_logvar =  (0.5 * (np.exp(logvar) - 1) + (d_decoder * 0.5 * np.exp(logvar)*self.eps))/ x.shape[0]
        self.encoder.backward(d_mu, d_logvar)

    def compute_vae_loss(self, x, x_reconstructed, mu, logvar):
        # VAE Loss
        reconstruction_loss = np.sum((x - x_reconstructed)**2) / x.shape[0]
        kl_divergence = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / x.shape[0]
        return reconstruction_loss, kl_divergence




    def train(self, x_train, batch_size=128, epochs=50):
        self.vae_loss_history, self.gan_loss_history, self.recon_loss_history, self.kl_loss_history = [], [], [], []

        for epoch in range(epochs):
            total_vae_loss, total_gan_loss = 0, 0
            recon_loss_, kl_loss = 0, 0

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                x_reconstructed, mu, logvar, z = self.forward(x_batch)

                # Compute losses
                recon_loss, kl = self.compute_vae_loss(x_batch, x_reconstructed, mu, logvar)
                total_vae_loss += recon_loss + kl
                recon_loss_ += recon_loss
                kl_loss += self.beta * kl

                # VAE Backward
                self.backward(x_batch, x_reconstructed, mu, logvar)

                # Update Encoder and Decoder
                for model in [self.encoder, self.decoder]:
                    model.update_params()


            # Logging
            avg_vae_loss = total_vae_loss / (len(x_train) / batch_size)
            avg_gan_loss = total_gan_loss / (len(x_train) / batch_size)
            self.vae_loss_history.append(avg_vae_loss)
            self.gan_loss_history.append(avg_gan_loss)
            self.recon_loss_history.append(recon_loss_ / (len(x_train) / batch_size))
            self.kl_loss_history.append(kl_loss / (len(x_train) / batch_size))

            print(f"Epoch {epoch + 1}/{epochs} - VAE Loss: {avg_vae_loss:.4f} - GAN Loss: {avg_gan_loss:.4f} - Recon Loss: {recon_loss_ / (len(x_train) / batch_size):.4f} - KL Loss: {kl_loss / (len(x_train) / batch_size):.4f}")



