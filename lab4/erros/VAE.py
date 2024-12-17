import numpy as np
import matplotlib.pyplot as plt


class VAE:
    def __init__(self, input_size, hidden_size, latent_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.learning_rate = learning_rate

        # Инициализация весов
        self.W_encoder = np.random.randn(input_size, hidden_size) * 0.01
        self.b_encoder = np.zeros((1, hidden_size))
        self.W_encoder_2 = np.random.randn(hidden_size, 128) * 0.01
        self.b_encoder_2 = np.zeros((1, 128))
        self.W_mu = np.random.randn(128, latent_size) * 0.01
        self.W_logvar = np.random.randn(128, latent_size) * 0.01
        self.W_decoder = np.random.randn(latent_size, hidden_size) * 0.01
        self.b_decoder = np.zeros((1, hidden_size))
        self.W_out = np.random.randn(hidden_size, input_size) * 0.01
        self.b_out = np.zeros((1, input_size))

        # инициализация гралиентов
        self.d_W_out = None
        self.d_out = None
        self.d_W_decoder = None
        self.d_z = None

        # вспомогательные параметры для backprop
        self.y = None
        self.h_decoder = None
        self.h_encoder_2 = None
        self.x_recon = None
        self.z = None
        self.h_encoder = None
        self.logvar = None
        self.mu = None
        self.eps = None
        self.std = None
        self.loss = None
        self.recon_loss = None
        self.kld_loss = None

    # Определение активационной функции
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    def encode(self, x):
        self.h_encoder = self.sigmoid(np.dot(x, self.W_encoder) + self.b_encoder)
        self.h_encoder_2 = self.sigmoid(np.dot(self.h_encoder, self.W_encoder_2) + self.b_encoder_2)

        # получаем с энкодера распределение признаков
        self.mu = np.dot(self.h_encoder_2, self.W_mu)
        self.logvar = np.dot(self.h_encoder_2, self.W_logvar)

    def reparametrize(self):
        # далее добавляем случайное распределение
        self.std = np.exp(0.5 * self.logvar)
        self.eps = np.random.normal(*self.mu.shape)
        self.z = self.mu + self.std * self.eps

    def decode(self):
        # декодировка
        self.h_decoder = self.sigmoid(np.dot(self.z, self.W_decoder) + self.b_decoder)
        self.x_recon = self.sigmoid(np.dot(self.h_decoder, self.W_out) + self.b_out)

    def compute_loss(self, x):
        recon_loss = np.mean(np.square(x - self.x_recon))  # средняя ошибка восстановления изображения
        kld_loss = -0.5 * np.sum(1 + self.logvar - self.mu ** 2 - np.exp(self.logvar))  # расстояние кульбака
        return recon_loss + kld_loss, recon_loss, kld_loss

    def backprop_decoder(self):
        # Обновление весов декодера
        d_rec = self.d_out * self.dsigmoid(self.x_recon)
        # вычисление градиента по выходным весам декодировщика полносвязного слоя
        self.d_W_out = np.dot(self.h_decoder.T, d_rec)

        # вычисление производной по картинке, получаемой с латентного пространства
        self.d_z = np.dot(self.d_out, self.W_out.T) * self.dsigmoid(self.h_decoder)
        # вычисление градиента по весам декодировщика, преобразующими картинку из латентного пространства
        self.d_W_decoder = np.dot(self.z.T, self.d_z)
        # вычисление производной по латентному пространству
        d_h = np.dot(self.d_z, self.W_decoder.T) * self.dsigmoid(self.z)
        return d_h

    def backprop_encoder(self, x, d_h):
        d_mu = d_h
        d_logvar = d_h * 0.5 * np.exp(self.logvar)
        d_mu += 0.0001 * self.mu
        d_logvar += 0.0001 * (-0.5 + 0.5 * np.exp(self.logvar))

        # вычисление градиентов по выборке
        d_W_mu = np.dot(self.h_encoder_2.T, d_mu)
        d_W_logvar = np.dot(self.h_encoder_2.T, d_logvar)

        # вычисление производной по распределению признаков
        d_h_encoder_2 = np.dot(d_mu, self.W_mu.T) + np.dot(d_logvar, self.W_logvar.T)
        d_h_encoder_2 *= self.dsigmoid(self.h_encoder_2)

        # вычисление производной по слою, преобразующему в латентное пространство
        d_W_encoder_2 = np.dot(self.h_encoder.T, d_h_encoder_2)

        # вычисление производной по распределению признаков
        d_h_encoder = np.dot(d_h_encoder_2, self.W_encoder_2.T) * self.dsigmoid(self.h_encoder)

        # вычисление производной по слою, преобразующему в латентное пространство
        d_W_encoder = np.dot(x.T, d_h_encoder)

        grads = {
            'w_out': self.d_W_out, 'b_out': self.d_out,
            'w_dec': self.d_W_decoder, 'b_dec': self.d_z,
            'w_enc': d_W_encoder, 'b_enc': d_h_encoder,
            'w_enc_2': d_W_encoder_2, 'b_enc_2': d_h_encoder_2,
            'w_mu': d_W_mu, 'w_logvar': d_W_logvar
        }
        return grads

    def train(self, X, epochs, batch_size, num):
        params = {
            'w_out': self.W_out, 'b_out': self.b_out,
            'w_dec': self.W_decoder, 'b_dec': self.b_decoder,
            'w_enc': self.W_encoder, 'b_enc': self.b_encoder,
            'w_enc_2': self.W_encoder_2, 'b_enc_2': self.b_encoder_2,
            'w_mu': self.W_mu, 'w_logvar': self.W_logvar
        }
        optimizer = AdamGD(lr=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, params=params)

        total = []
        total_kld = []
        for epoch in range(epochs):
            total_loss = []
            total_kld_loss = []
            for _ in range(0, len(X), batch_size):
                X_batch = np.array(X[_:_ + batch_size])

                # Прямое распространение
                self.encode(X_batch)
                self.reparametrize()
                self.decode()

                # Вычисление функции потерь
                loss, recon_loss, kld_loss = self.compute_loss(X_batch)
                total_loss.append(recon_loss)
                total_kld_loss.append(kld_loss)

                # вычисление производной по лоссу
                self.d_out = (2 / X_batch.shape[0]) * (self.x_recon - X_batch)

                # обратный проход
                d_h = self.backprop_decoder()
                grads = self.backprop_encoder(X_batch, d_h)

                # обновление весов
                params = optimizer.update_params(grads)
                self.set_params(params)

            if epoch == epochs-1:
                for _ in range(0, 50, batch_size):
                    X_batch = np.array(X[_:_ + batch_size])
                    for i in range(1):
                        image = X_batch[i].reshape(28, 28) * 255
                        fake = self.x_recon[i].reshape(28, 28) * 255

                        image_fake = np.hstack([image, fake])

                        fig = plt.figure(figsize=(6, 4))
                        ax = fig.add_subplot()
                        ax.imshow(image_fake)
                        plt.savefig(rf'C:\Users\makso\Desktop\ФООСИИ\vae_gan\Картинки\fig{num}_{i}.png')

            loss_avg = np.mean(total_loss)
            kld_loss_avg = np.mean(total_kld_loss)
            total.append(loss_avg)
            total_kld.append(kld_loss_avg)
            print(f'MSE Loss: {loss_avg:.5f}, KLD Loss: {kld_loss_avg:.5f}, Epoch: {epoch}')

        return total, total_kld

    def generate(self, X_batch):
        # Прямое распространение
        self.encode(X_batch)
        self.reparametrize()
        self.decode()

        # Вычисление функции потерь
        self.loss, self.recon_loss, self.kld_loss = self.compute_loss(X_batch)
        return self.x_recon

    def backward(self, X_batch, d_disc):
        params = {
            'w_out': self.W_out, 'b_out': self.b_out,
            'w_dec': self.W_decoder, 'b_dec': self.b_decoder,
            'w_enc': self.W_encoder, 'b_enc': self.b_encoder,
            'w_enc_2': self.W_encoder_2, 'b_enc_2': self.b_encoder_2,
            'w_mu': self.W_mu, 'w_logvar': self.W_logvar
        }
        optimizer = AdamGD(lr=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, params=params)

        self.d_out = (2 / X_batch.shape[0]) * (self.x_recon - X_batch) - d_disc
        d_h = self.backprop_decoder()
        grads = self.backprop_encoder(X_batch, d_h)

        # обновление весов
        params = optimizer.update_params(grads)
        self.set_params(params)

        return self.recon_loss, self.kld_loss

    def set_params(self, params):
        self.W_out = params['w_out']
        self.b_out = params['b_out']

        self.W_encoder = params['w_enc']
        self.b_encoder = params['b_enc']

        self.W_encoder_2 = params['w_enc_2']
        self.b_encoder_2 = params['b_enc_2']

        self.W_decoder = params['w_dec']
        self.b_decoder = params['b_dec']

        self.W_mu = params['w_mu']
        self.W_logvar = params['w_logvar']


class AdamGD:
    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params

        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = np.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):
        for key in self.params:
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads[key]
            self.rmsprop['sd' + key] = (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (grads[key] ** 2)
            self.params[key] = self.params[key] - (self.lr * self.momentum['vd' + key]) / (np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)
        return self.params


class Relu:
    def __init__(self):
        self.y = None

    def relu(self, x):
        # вычисление значения
        y = np.where(x > 0, x, 0)
        self.y = y  # для бэкпропа
        return y

    def drelu(self, dE_dy):
        # бэкпроп
        dy_dx = np.where(self.y <= 0, self.y, 1)
        dE_dx = dE_dy * dy_dx
        return dE_dx
