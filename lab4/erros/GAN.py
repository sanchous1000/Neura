import numpy as np
import matplotlib.pyplot as plt


class GAN:
    def __init__(self, input_size, hidden_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.W_generator = np.random.randn(hidden_size, input_size) * 0.01
        self.b_generator = np.zeros((1, input_size))

        self.W_discriminator = np.random.randn(input_size, hidden_size) * 0.01
        self.b_discriminator = np.zeros((1, hidden_size))

        self.W_out_discriminator = np.random.randn(hidden_size, 1) * 0.01
        self.b_out_discriminator = np.zeros((1, 1))

        self.disc_real = None
        self.disc_fake = None
        self.h_discriminator = None

    # Определение активационной функции
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    def generate(self, noise):
        return self.sigmoid(np.dot(noise, self.W_generator) + self.b_generator)

    def discriminate(self, x):
        self.h_discriminator = self.sigmoid(np.dot(x, self.W_discriminator) + self.b_discriminator)
        return self.sigmoid(np.dot(self.h_discriminator, self.W_out_discriminator) + self.b_out_discriminator)

    def train(self, X, epochs, batch_size, vae, num):
        total = []
        total_kld = []
        total_gan = []

        for epoch in range(epochs):
            total_loss = []
            total_kld_loss = []
            total_gan_loss = []

            for _ in range(0, len(X), batch_size):
                real_batch = np.array(X[_:_ + batch_size])

                # Генерируем шум
                fake_data = vae.generate(real_batch)

                self.disc_real = self.discriminate(real_batch)
                self.disc_fake = self.discriminate(fake_data)

                mse_loss, kld_loss, gan_loss = self.backprop(real_batch, fake_data, vae)

                total_loss.append(mse_loss)
                total_kld_loss.append(kld_loss)
                total_gan_loss.append(gan_loss)

            if epoch == epochs-1:
                for _ in range(0, 50, batch_size):
                    X_batch = np.array(X[_:_ + batch_size])
                    for i in range(1):
                        image = X_batch[i].reshape(28, 28) * 255
                        fake = fake_data[i].reshape(28, 28) * 255

                        image_fake = np.hstack([image, fake])

                        fig = plt.figure(figsize=(6, 4))
                        ax = fig.add_subplot()
                        ax.imshow(image_fake)
                        plt.savefig(rf'C:\Users\makso\Desktop\ФООСИИ\vae_gan\Картинки\fig_{num}_{i}{i}.png')

            loss_avg = np.mean(total_loss)
            kld_loss_avg = np.mean(total_kld_loss)
            gan_loss_avg = np.mean(total_gan_loss)

            total.append(loss_avg)
            total_kld.append(kld_loss_avg)
            total_gan.append(gan_loss_avg)
            print(f'MSE Loss: {loss_avg:.5f}, KLD Loss: {kld_loss_avg:.5f}, GAN Loss: {gan_loss_avg:.5f}, Epoch: {epoch}')
        return total, total_kld, total_gan

    def backprop(self, real_batch, fake_data, vae):
        # Обновляем дискриминатор
        d_loss = -np.sum(self.disc_real * np.log(self.disc_fake)) / self.disc_fake.shape[0]

        d_out = self.disc_real - self.disc_fake
        d_W_discriminator_out = np.dot(self.h_discriminator.T, d_out)

        d_discriminator = np.dot(d_out, self.W_out_discriminator.T) * self.dsigmoid(self.h_discriminator)
        d_W_discriminator = np.dot(real_batch.T, d_discriminator)

        self.W_out_discriminator -= self.learning_rate * d_W_discriminator_out
        self.b_out_discriminator -= self.learning_rate * np.sum(d_out, axis=0, keepdims=True)

        self.W_discriminator -= self.learning_rate * d_W_discriminator
        self.b_discriminator -= self.learning_rate * np.sum(d_discriminator, axis=0, keepdims=True)

        # Обновление весов генератора
        d_discriminator = np.dot(d_out, self.W_out_discriminator.T) * self.dsigmoid(self.h_discriminator)
        d_discriminator_2 = np.dot(d_discriminator, self.W_discriminator.T) * self.dsigmoid(fake_data)

        mse_loss, kld_loss = vae.backward(real_batch, d_discriminator_2)

        return mse_loss, kld_loss, d_loss
