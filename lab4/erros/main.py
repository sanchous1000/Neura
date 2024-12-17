import matplotlib.pyplot as plt
import os
import shutil

from Loader import Loader
from VAE import VAE
from GAN import GAN

loader = Loader()

dict_nums = loader.dict_nums_train

input_size = 784
hidden_size = 256
latent_size = 50

dict_vae = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for num, X_train in dict_nums.items():
    vae = VAE(input_size, hidden_size, latent_size, 1e-3)
    loss_mse_1, loss_kld_1 = vae.train(X=X_train, epochs=3, batch_size=5, num=num)

    gan = GAN(input_size, hidden_size, learning_rate=1e-4)
    loss_mse_2, loss_kld_2, loss_gan_2 = gan.train(X=X_train, batch_size=5, epochs=15, vae=vae, num=num)

    dict_vae[num] = vae

    try:
        shutil.rmtree(rf'C:\Users\makso\Desktop\ФООСИИ\vae_gan\Лосс\{num}')
    except FileNotFoundError:
        pass

    os.makedirs(rf'C:\Users\makso\Desktop\ФООСИИ\vae_gan\Лосс\{num}', exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot([_ for _ in range(len(loss_mse_1))], loss_mse_1)
    loss_mse_2.insert(0, loss_mse_1[-1])
    plt.plot([_+len(loss_mse_1)-1 for _ in range(len(loss_mse_2))], loss_mse_2)
    plt.savefig(rf'C:\Users\makso\Desktop\ФООСИИ\vae_gan\Лосс\{num}\MSE.png')

    plt.figure(figsize=(6, 4))
    plt.plot([_ for _ in range(len(loss_kld_1))], loss_kld_1)
    loss_kld_2.insert(0, loss_kld_1[-1])
    plt.plot([_+len(loss_kld_1)-1 for _ in range(len(loss_kld_2))], loss_kld_2)
    plt.savefig(rf'C:\Users\makso\Desktop\ФООСИИ\vae_gan\Лосс\{num}\KLD.png')

    plt.figure(figsize=(6, 4))
    plt.plot(loss_gan_2)
    plt.savefig(rf'C:\Users\makso\Desktop\ФООСИИ\vae_gan\Лосс\{num}\GAN.png')
    print('____________________')
