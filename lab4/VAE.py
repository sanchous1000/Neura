import numpy as np
from main_cl import Fullyconnected, Flatten, Conv
from activations import LeakyRelu, Tanh, Sigmoid

import numpy as np

import numpy as np

def binary_cross_entropy(input, target, weight=None, reduction='sum'):
    # Предотвращение логарифма нуля
    epsilon = 1e-12
    input = np.clip(input, epsilon, 1 - epsilon)
    
    # Формула для binary cross-entropy
    loss = - (target * np.log(input) + (1 - target) * np.log(1 - input))
    
    # Применение весов, если указано
    if weight is not None:
        loss *= weight
    
    # Применение редукции
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Неверный тип редукции: {reduction}")




class Encoder:
    def __init__(self, arch = (784, 200, 50), zdims=2):
        self.zdims = zdims
        self.flatten = Flatten()
        self.fc1 = Fullyconnected(input_size=arch[0], output_size=arch[1])
        self.LeakyRelu1 =  LeakyRelu()
        self.fc2 = Fullyconnected(input_size=arch[1], output_size=arch[2])
        self.LeakyRelu2 =  LeakyRelu()
        self.mu = Fullyconnected(input_size=arch[2], output_size=zdims)
        self.log_var = Fullyconnected(input_size=arch[2], output_size=zdims)


        self.layers = [self.flatten, 
                       self.fc1,
                     self.LeakyRelu1,
                     self.fc2,
                     self.LeakyRelu2,
                     [self.mu,
                      self.log_var]]

        
        

        '''self.en_x_1 = Fullyconnected(input_size=input_dim, output_size=hidden_dim)
        self.LeakyRelu1 = LeakyRelu()
        self.en_x_2 = Fullyconnected(input_size=hidden_dim, output_size=hidden_dim)
        self.LeakyRelu2 = LeakyRelu()
        self.en_x_4_mu = Fullyconnected(input_size=hidden_dim, output_size=zdims)
        self.en_x_4_sigma = Fullyconnected(input_size=hidden_dim, output_size=zdims)'''

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
        '''dh = self.en_x_4_mu.backward(dmu) + self.en_x_4_sigma.backward(dlogvar)
         # Прямой проход
        h = x'''
        
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
    
    def update_params(self,lr):
        for layer in self.layers:
            if 'activations' not in str(layer.__class__):
                if 'Flatten' not in str(layer.__class__):
                    if type(layer) == list:
                        layer[0].update_params(lr)
                        layer[1].update_params(lr)
                    else:
                        layer.update_params(lr)
            

class Decoder:
    def __init__(self, zdims=2, arch = (784, 200, 50)):
    
        self.de_x_1 = Fullyconnected(input_size=zdims, output_size=arch[2])
        self.LeakyRelu1 = LeakyRelu()
        self.de_x_2 = Fullyconnected(input_size=arch[2], output_size=arch[1])
        self.LeakyRelu2 = LeakyRelu()
        self.de_x_3 = Fullyconnected(input_size=arch[1], output_size=arch[0])
        self.sig = Sigmoid()
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
    
    def update_params(self, lr):
        for layer in self.layers:
             if 'activations' not in str(layer.__class__):
                if 'Flatten' not in str(layer.__class__):
                    layer.update_params(lr)
        



class Discriminator:
    def __init__(self, input_dim=784, hidden_dim1=1024, hidden_dim2=512, hidden_dim3=256):
        
        
        # Полносвязные слои для кодировщика
        self.en_x_1 = Fullyconnected(input_size=input_dim, output_size=hidden_dim1)
        self.LeakyRelu1 = LeakyRelu()
        self.en_x_2 = Fullyconnected(input_size=hidden_dim1, output_size=hidden_dim2)
        self.LeakyRelu2 = LeakyRelu()
        self.en_x_3 = Fullyconnected(input_size=hidden_dim2, output_size=hidden_dim3)
        self.LeakyRelu3 = LeakyRelu()
        self.en_x_4 = Fullyconnected(input_size=hidden_dim3, output_size=1)
        self.sig = Sigmoid()

        

    def forward(self, x):
      
        h = self.LeakyRelu1.forward(self.en_x_1.forward(x))
        h = self.LeakyRelu2.forward(self.en_x_2.forward(h))
        h = self.LeakyRelu3.forward(self.en_x_3.forward(h))   
        return self.sig.forward(self.en_x_4.forward(h))

    def backward(self, dout):
      
        dh = self.sig.backward(dout)
        dh = self.en_x_4.backward(dh) 
        dh = self.LeakyRelu3.backward(dh)
        dh = self.en_x_3.backward(dh)

        dh = self.LeakyRelu2.backward(dh)
        dh = self.en_x_2.backward(dh)

        dh = self.LeakyRelu1.backward(dh)
        dh = self.en_x_1.backward(dh)
       
        return dh
    def update_params(self, lr):
        self.en_x_1.update_params(lr)
        self.en_x_2.update_params(lr)
        self.en_x_3.update_params(lr)
        self.en_x_4.update_params(lr)

        


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class VAEGANTrainer:
    def __init__(self, encoder, decoder, learning_rate=1e-4):
        self.encoder = encoder
        self.decoder = decoder
        #self.discriminator = discriminator
        self.learning_rate = learning_rate

    def train(self, x_train, batch_size=128, epochs=50):
        vae_loss_history = []
        gan_loss_history = []
        kl_loss_history = []

        for epoch in range(epochs):
            total_vae_loss = 0
            total_gan_loss = 0
            total_kl_loss = 0

            for i in range(0, len(x_train), batch_size):
                # Формируем батч
                x_batch = x_train[i:i + batch_size]
                batch_size_actual = x_batch.shape[0]

                # 1. Прямой проход через VAE
                mu, logvar = self.encoder.forward(x_batch)
                z = self.encoder.reparameterize(mu, logvar)
                x_reconstructed = self.decoder.forward(z)
                
                recon_loss = binary_cross_entropy(x_reconstructed,x_batch)
                
                kl_loss = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar)) / batch_size_actual
                vae_loss = recon_loss + kl_loss


                d_recon = (x_reconstructed - x_batch)
                d_decoder = self.decoder.backward(d_recon)

    
                dmu = d_decoder + mu  
                dlogvar = d_decoder * (z - mu) * 0.5 * np.exp(logvar)  # Градиенты по logvar

                # Обратное распространение через кодировщик
                self.encoder.backward(dmu, dlogvar)
                            

                

                # Сохранение метрик
                total_vae_loss += vae_loss
                #total_gan_loss += gen_loss
                total_kl_loss += kl_loss

                # Обновление параметров
                for model in [self.encoder, self.decoder]:
                    model.update_params(self.learning_rate)

            # Сохранение истории
            vae_loss_history.append(total_vae_loss / len(x_train))
            #gan_loss_history.append(total_gan_loss / len(x_train))
            kl_loss_history.append(total_kl_loss / len(x_train))

            # Вывод метрик
            print(f"Epoch {epoch + 1}/{epochs} - VAE Loss: {total_vae_loss:.4f}, GAN Loss: {total_gan_loss:.4f}, KL Loss: {total_kl_loss:.4f}")

        # Построение графиков
        self.plot_metrics(vae_loss_history, gan_loss_history, kl_loss_history)

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
