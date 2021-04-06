import numpy as np
import pandas as pd
from keras.models import Sequential, Model, Input
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Nadam
from sklearn.model_selection import train_test_split
from tools import save_info, show_plot
from tools import mean_absolute_percentage_error
from scipy.stats.stats import pearsonr
import time
from sklearn.preprocessing import MinMaxScaler
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Bidirectional


np.random.seed(2020)


class DNN(object):
    def __init__(self, bs, ts):
        self.NB_EPOCH = 2000
        self.BATCH_SIZE = 1000
        self.VERBOSE = 1
        self.OPTIMIZER = Adam()
        self.N_HIDDEN = 512
        self.VALIDATION_SPLIT = 0.14
        self.INPUT_DIM = int(bs)
        self.bs = bs
        self.ts = ts
        self.WINDOW_SIZE = int(bs)
        self.WINDOW_SIZE = int(bs)

    def prepare(self, X):
        X, Y = self.prepare_data(X)
        Y = np.array(Y)

        Y = Y.reshape(-1, 1)

        return X, Y

    def normalize(self, X, Y):
        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(X)
        X = scaler_X.transform(X)

        scaler_Y = MinMaxScaler()
        scaler_Y = scaler_Y.fit(Y)
        Y = scaler_Y.transform(Y)

        X = pd.DataFrame(X)

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        return X, Y

    def split(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, shuffle=False)

        self.X_train = X_train
        self.y_train = Y_train
        self.X_test = X_test
        self.y_test = Y_test

    def prepare_data(self, X):
        series = pd.Series(X)
        series_s = series.copy()

        for i in range(self.WINDOW_SIZE):
            series = pd.concat([series, series_s.shift(-(i+1))], axis=1)
        # series.dropna(axis=0, inplace=True)
        series.columns = np.arange(self.WINDOW_SIZE + 1)

        X_new = pd.DataFrame()
        for i in range(self.WINDOW_SIZE):
            X_new[i] = series[i]
        Y_new = series[self.WINDOW_SIZE]

        return X_new, Y_new

    def dnn(self, path=None, name=None):
        with tf.device('/device:GPU:0'):
            model = Sequential()
            # print(self.X_train.shape[1])
            model.add(Dense(self.X_train.shape[1], input_shape=(self.X_train.shape[1],)))
            model.add(Activation('relu'))
            for i in range(3):
                model.add(Dense(self.N_HIDDEN))
                model.add(Activation('relu'))
            model.add(Dense(1))
            model.add(Activation('soft-max'))
            model.summary()
            model.compile(loss='mse',
                          optimizer=self.OPTIMIZER,
                          metrics=['accuracy'])
            history = model.fit(self.X_train, self.y_train,
                                epochs=self.NB_EPOCH,
                                verbose=self.VERBOSE)
            print(self.X_train)

            y_pred = model.predict(self.X_test)

            y_pred = y_pred.reshape(-1)

            plt.plot(self.y_test)
            plt.plot(y_pred)
            plt.legend(['real', 'prediction'])
            plt.savefig(f'./results/{name}.png')
            plt.clf()
            # plt.show()

            mse = MeanSquaredError()
            loss_mse = mse(self.y_test, y_pred).numpy()

            loss_rmse = np.sqrt(loss_mse)

            mae = MeanAbsoluteError()
            loss_mae = mae(self.y_test, y_pred).numpy()

            mape = MeanAbsolutePercentageError()
            loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mse, loss_mae, loss_mape

    def lstm(self, path=None, name=None):
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        trainX = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        testX = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        model = Sequential()
        model.add(LSTM(128, batch_input_shape=(1, trainX.shape[1], trainX.shape[2]), stateful=True))
        # model.add(Activation('tanh'))
        model.add(Dense(1))
        # model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()

        for i in range(2000):
            model.fit(trainX, self.y_train, epochs=1, batch_size=1, verbose=self.VERBOSE, shuffle=False)
            model.reset_states()

        y_pred = model.predict(testX, batch_size=1)

        y_pred = y_pred.reshape(-1)

        plt.plot(self.y_test)
        plt.plot(y_pred)
        plt.legend(['real', 'prediction'])
        plt.savefig(f'./results/{name}.png')
        plt.clf()
        # plt.show()

        mse = MeanSquaredError()
        loss_mse = mse(self.y_test, y_pred).numpy()

        loss_rmse = np.sqrt(loss_mse)

        mae = MeanAbsoluteError()
        loss_mae = mae(self.y_test, y_pred).numpy()

        mape = MeanAbsolutePercentageError()
        loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mse, loss_mae, loss_mape

    def lstm_with_sequence(self, path=None, name=None):
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values

        trainX = np.reshape(self.X_train, (self.X_train.shape[0], -1, 3))
        testX = np.reshape(self.X_test, (self.X_test.shape[0], -1, 3))

        with tf.device('/device:CPU:0'):
            model = Sequential()
            model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            # model.summary()

            model.fit(trainX, self.y_train, epochs=300, batch_size=10, verbose=self.VERBOSE, shuffle=False)

            y_pred = model.predict(testX, batch_size=1)
            y_pred = y_pred.reshape(-1)

            print(self.y_test.shape)
            print(y_pred.shape)

            plt.plot(self.y_test)
            plt.plot(y_pred)
            plt.legend(['real', 'prediction'])
            plt.savefig(f'./results/{name}.png')
            plt.clf()
            # plt.show()

            mse = MeanSquaredError()
            loss_mse = mse(self.y_test, y_pred).numpy()

            loss_rmse = np.sqrt(loss_mse)

            mae = MeanAbsoluteError()
            loss_mae = mae(self.y_test, y_pred).numpy()

            mape = MeanAbsolutePercentageError()
            loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mse, loss_mae, loss_mape

    def svr(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        regr = make_pipeline(SVR(C=1.0, epsilon=0.2))
        regr.fit(self.X_train, self.y_train)

        y_pred = regr.predict(self.X_test)

        y_pred = y_pred.reshape(-1)

        mse = MeanSquaredError()
        loss_mse = mse(self.y_test, y_pred).numpy()

        loss_rmse = np.sqrt(loss_mse)

        mae = MeanAbsoluteError()
        loss_mae = mae(self.y_test, y_pred).numpy()

        mape = MeanAbsolutePercentageError()
        loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mae, loss_mape

    @staticmethod
    def ENN():
        import neurolab as nl

        # Create train samples
        i1 = np.sin(np.arange(0, 20))
        i2 = np.sin(np.arange(0, 20)) * 2

        t1 = np.ones([1, 20])
        t2 = np.ones([1, 20]) * 2

        input = np.array([i1, i2, i1, i2]).reshape(20 * 4, 1)
        target = np.array([t1, t2, t1, t2]).reshape(20 * 4, 1)

        # Create network with 2 layers
        net = nl.net.newelm([[-1, 1]], [50, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        # Set initialized functions and init
        net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
        net.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
        net.init()
        # Train network
        error = net.train(input, target, epochs=500, show=100, goal=0.01)
        # Simulate network
        output = net.sim(input)
        print(output)

        # Plot result
        import pylab as pl
        pl.subplot(211)
        pl.plot(error)
        pl.xlabel('Epoch number')
        pl.ylabel('Train error (default MSE)')

        pl.subplot(212)
        pl.plot(target.reshape(80))
        pl.plot(output.reshape(80))
        pl.legend(['train target', 'net output'])
        pl.show()



        # y_pred = regr.predict(self.X_test)
        #
        # y_pred = y_pred.reshape(-1)
        #
        # mse = MeanSquaredError()
        # loss_mse = mse(self.y_test, y_pred).numpy()
        #
        # loss_rmse = np.sqrt(loss_mse)
        #
        # mae = MeanAbsoluteError()
        # loss_mae = mae(self.y_test, y_pred).numpy()
        #
        # mape = MeanAbsolutePercentageError()
        # loss_mape = mape(self.y_test, y_pred).numpy()
        #
        # return loss_rmse, loss_mae, loss_mape

    def CNN(self, name=None):
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values

        trainX = np.reshape(self.X_train, (self.X_train.shape[0], 9, -1))
        testX = np.reshape(self.X_test, (self.X_test.shape[0], 9, -1))

        with tf.device('/device:CPU:0'):
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Dropout(0.5))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(100, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mse', optimizer='adam')

            model.fit(trainX, self.y_train, epochs=2000, batch_size=32, verbose=self.VERBOSE)

            y_pred = model.predict(testX, batch_size=1)
            y_pred = y_pred.reshape(-1)

            print(self.y_test.shape)
            print(y_pred.shape)

            plt.plot(self.y_test)
            plt.plot(y_pred)
            plt.legend(['real', 'prediction'])
            plt.savefig(f'./results/{name}.png')
            plt.clf()
            # plt.show()

            mse = MeanSquaredError()
            loss_mse = mse(self.y_test, y_pred).numpy()

            loss_rmse = np.sqrt(loss_mse)

            mae = MeanAbsoluteError()
            loss_mae = mae(self.y_test, y_pred).numpy()

            mape = MeanAbsolutePercentageError()
            loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mse, loss_mae, loss_mape

    def CNN_LSTM(self, name=None):
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values

        trainX = np.reshape(self.X_train, (self.X_train.shape[0], 64, -1))
        testX = np.reshape(self.X_test, (self.X_test.shape[0], 64, -1))

        with tf.device('/device:CPU:0'):
            model = Sequential()
            input_layer = Input(shape=(64, 1))
            conv1 = Conv1D(filters=32, kernel_size=8, strides=1, activation='relu', padding='same')(input_layer)
            lstm1 = LSTM(32, return_sequences=True)(conv1)
            output_layer = Dense(1, activation='linear')(lstm1)
            model = Model(inputs=input_layer, outputs=output_layer)
            # print(model.summary())
            model.compile(loss='mse', optimizer='adam')

            model.fit(trainX, self.y_train, epochs=2000, batch_size=32, verbose=self.VERBOSE)

            y_pred = model.predict(testX, batch_size=1)
            y_pred = y_pred.reshape(-1)

            print(self.y_test.shape)
            print(y_pred.shape)

            plt.plot(self.y_test)
            plt.plot(y_pred)
            plt.legend(['real', 'prediction'])
            plt.savefig(f'./results/{name}.png')
            plt.clf()
            # plt.show()

            mse = MeanSquaredError()
            loss_mse = mse(self.y_test, y_pred).numpy()

            loss_rmse = np.sqrt(loss_mse)

            mae = MeanAbsoluteError()
            loss_mae = mae(self.y_test, y_pred).numpy()

            mape = MeanAbsolutePercentageError()
            loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mse, loss_mae, loss_mape

#     def calculate_corr(self):
#         # corr = pearsonr(self.X.T, self.Y.T)
#         # np.savetxt('corrdata.txt', corr)
#         np.savetxt('X.txt', self.X)
#         np.savetxt('Y.txt', self.Y)

