import numpy as np
from activations import LeakyRelu, Tanh, Sigmoid, Relu
rng = np.random.default_rng(51)
import numpy as np



import matplotlib.pyplot as plt
def visualize_reconstruction(decoder,encoder , test_data, num_images=10):
    # Выберите случайные изображения
    indices = np.random.choice(len(test_data), num_images)
    original_images = test_data[indices]
    reconstructed_images = [decoder.forward(encoder.forward(img.reshape(1, -1))[0]) for img in original_images]

    # Построение графика
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        # Оригинал
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title("Original")
        # Реконструкция
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title("Reconstructed")
    plt.show()

class Fullyconnected: 
    def __init__(self, input_size, output_size, derivative = False ):
        self.derivative = derivative
        #
        self.W = self._init_weights(input_size, output_size)
        self.b = self._init_biases(output_size)
        #opt_params
        self.t = 1
        self.mW =np.zeros_like(self.W) 
        self.mb = np.zeros_like(self.b) 
        self.vW =np.zeros_like(self.W) 
        self.vb = np.zeros_like(self.b) 

    def forward(self, X):
        self.a_l = X
        z = np.dot(X, self.W.T) + self.b
        return z 

    def backward(self, dout):
        m =  self.a_l.shape[0]
        self.dW = np.dot(dout.T, self.a_l) / m 
        self.db = np.sum(dout, axis = 0, keepdims=True) / m
        delta = np.dot(dout, self.W)
        return delta
    
    def backward_(self, dout):
        m =  self.a_l.shape[0]
        delta = np.dot(dout, self.W)
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
        self.W -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
        self.b  -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        self.t += 1
    def _init_weights(self, input_size, output_size):
        net_in = input_size
        net_out = output_size
        limit = np.sqrt(6. / (net_in + net_out))
        return rng.uniform(-limit, limit + 1e-5, size=(net_out, net_in)) 
    #
    def _init_biases(self, output_size):
        return np.zeros((1,output_size)) 
    
    


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
        

from activations import LeakyRelu, Tanh, Sigmoid, Relu

class Discriminator:
    def __init__(self, input_size=784, arch=(256, 128), learning_rate=1e-8):
        self.fc1 = Fullyconnected(input_size=input_size, output_size=arch[0])
        #self.LeakyRelu1 = Tanh()
        self.fc2 = Fullyconnected(input_size=arch[0], output_size=arch[1])
       # self.LeakyRelu2 = Tanh()
        self.fc3 = Fullyconnected(input_size=arch[1], output_size=1)
        #self.LeakyRelu3 = Relu()
        #self.fc4 = Fullyconnected(input_size=arch[1], output_size=1)
        self.sig = Sigmoid()
        self.lr = learning_rate

        self.layers = [
            self.fc1,
            #self.LeakyRelu1,
            self.fc2,
            #self.LeakyRelu2,
            self.fc3,
           # self.LeakyRelu3,
           # self.fc4,
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


        

class GAN_:
    def __init__(self, generator, vae, discriminator, beta=1, dec_lam=1):
        self.vae = vae
        self.encoder = generator
        self.discriminator = discriminator

    def forward(self, x):
        recon,  mu, logvar, recon_loss, kl = self.vae.forward(x)
        real_preds = self.discriminator.forward(x)
        fake_preds = self.discriminator.forward(recon)
        return real_preds, recon,  fake_preds, mu, logvar,  recon_loss, kl

    def binary_l(self, predictions, targets):
        """Бинарная кросс-энтропия."""
        predictions = np.clip(predictions, 1e-8, 1 - 1e-8)  # Для стабильности
        grad = (predictions - targets) / (predictions * (1 - predictions))
        grad /= targets.shape[0]  # Нормализация по числу примеров в батче
        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return loss, grad

    def train(self, x_train, batch_size=128, epochs=50):
        self.total_d_loss, self.total_g_loss = [], []
        self.total_recon_loss, self.total_kl_loss = [], []


        for epoch in range(epochs):
            total_d_loss, total_g_loss,total_recon_loss, total_kl_loss  = 0, 0, 0,0

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]

                '''real_labels = np.ones((x_batch.shape[0], 1))
                fake_labels = np.zeros((x_batch.shape[0], 1))'''
                
                real_preds, recon, fake_preds, mu, logvar, recon_loss, kl = self.forward(x_batch)
                
                total_recon_loss += recon_loss 
                total_kl_loss += kl


                d_loss = -(np.sum( real_preds * np.log(fake_preds +  1e-8)) ) / x_batch.shape[0]
                total_g_loss += d_loss
                d_out = real_preds - fake_preds             
                
                #g_loss = -np.mean(np.log(fake_preds + 1e-8))
                
                grad_g_loss = self.discriminator.backward_(d_out / x_batch.shape[0])
                self.vae.backward(x_batch, recon, mu, logvar, grad_g_loss)
                self.discriminator.backward(d_out)
                self.discriminator.update_params()

                


            # === Logging ===
            visualize_reconstruction(decoder=self.vae.decoder, encoder=self.vae.encoder, test_data=x_train)
            avg_d_loss = total_d_loss  / (len(x_train) / batch_size)
            avg_g_loss = total_g_loss / (len(x_train) / batch_size)
            avg_kl_loss = total_kl_loss / (len(x_train) / batch_size)
            avg_recon_loss  = total_recon_loss/ (len(x_train) / batch_size)
            self.total_d_loss.append(avg_d_loss)
            self.total_g_loss.append(avg_g_loss)
            self.total_recon_loss.append(avg_recon_loss)
            self.total_kl_loss.append(avg_kl_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} - D Loss: {avg_d_loss:.4f} - G Loss: {avg_g_loss:.4f} - Recon: {avg_recon_loss:.4f} - Kl loss: {avg_kl_loss:.4f} ")
        


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class VAEGANTrainer:
    def __init__(self, encoder, decoder, beta = 1, dec_lam = 1):
        self.encoder = encoder
        self.decoder = decoder
        self.dec_lam = dec_lam
        self.beta = beta

    def forward(self, x):
        mu, logvar = self.encoder.forward(x)
        z , self.eps = self.encoder.reparameterize(mu, logvar)
        x_reconstructed = self.decoder.forward(z)
        recon_loss, kl = self.compute_vae_loss(x, x_reconstructed, mu, logvar)
                
        return x_reconstructed, mu, logvar, recon_loss, kl

    def backward(self, x, x_reconstructed, mu, logvar, d_gan = 0):
        d_reconstruction =  self.dec_lam * (2 * (x_reconstructed - x) / x.shape[0]) - d_gan
        d_decoder = self.decoder.backward(d_reconstruction / x.shape[0])
        d_mu = (d_decoder +  self.beta * mu) / x.shape[0] #(d_decoder +   self.beta * mu) / x.shape[0]
        d_logvar = self.beta * (d_decoder * 0.5 * np.exp(logvar)*self.eps)/ x.shape[0]
        self.encoder.backward(d_mu, d_logvar)
        
        for model in [self.encoder, self.decoder]:
            model.update_params()
       


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
                x_reconstructed, mu, logvar, recon_loss, kl  = self.forward(x_batch)

                # Compute losses
                total_vae_loss += recon_loss + kl
                recon_loss_ += recon_loss
                kl_loss += self.beta * kl

                # VAE Backward
                self.backward(x_batch, x_reconstructed, mu, logvar)


            # Logging
            visualize_reconstruction(decoder=self.decoder, encoder=self.encoder, test_data=x_train)
            avg_vae_loss = total_vae_loss / (len(x_train) / batch_size)
            avg_gan_loss = total_gan_loss / (len(x_train) / batch_size)
            self.vae_loss_history.append(avg_vae_loss)
            self.gan_loss_history.append(avg_gan_loss)
            self.recon_loss_history.append(recon_loss_ / (len(x_train) / batch_size))
            self.kl_loss_history.append(kl_loss / (len(x_train) / batch_size))

            print(f"Epoch {epoch + 1}/{epochs} - VAE Loss: {avg_vae_loss:.4f} - GAN Loss: {avg_gan_loss:.4f} - Recon Loss: {recon_loss_ / (len(x_train) / batch_size):.4f} - KL Loss: {kl_loss / (len(x_train) / batch_size):.4f}")



