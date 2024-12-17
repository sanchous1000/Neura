import numpy as np
import pandas as pd


class Loader:
    def __init__(self):
        x_train, y_train, x_test, y_test = self.load_data('3. mnist.npz')
        self.x_train = x_train.reshape(-1, 28*28) / 255
        self.x_test = x_train.reshape(-1, 28 * 28) / 255
        self.x_train = self.x_train[:3000]
        self.x_test = self.x_test[:1000]

        self.dict_nums_train = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }

        self.dict_nums_test = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }

        self.y_train = pd.Series(y_train[:3000])
        for num in self.y_train.unique():
            self.dict_nums_train[num] = self.x_train[list(self.y_train.loc[self.y_train == num].index)]
        self.dict_nums_train[0] = self.dict_nums_train[0][:280]
        self.dict_nums_train[1] = self.dict_nums_train[1][:330]
        self.dict_nums_train[2] = self.dict_nums_train[2][:290]
        self.dict_nums_train[3] = self.dict_nums_train[3][:290]
        self.dict_nums_train[4] = self.dict_nums_train[4][:320]
        self.dict_nums_train[5] = self.dict_nums_train[5][:270]
        self.dict_nums_train[6] = self.dict_nums_train[6][:300]
        self.dict_nums_train[7] = self.dict_nums_train[7][:320]
        self.dict_nums_train[8] = self.dict_nums_train[8][:260]
        self.dict_nums_train[9] = self.dict_nums_train[9][:280]

        self.y_test = pd.Series(y_test[:1000])
        for num in self.y_test.unique():
            self.dict_nums_test[num] = self.x_test[list(self.y_test.loc[self.y_test == num].index)]

    # загрузка данных
    @staticmethod
    def load_data(path):
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
            return x_train, y_train, x_test, y_test



