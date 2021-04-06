import numpy as np
import pandas as pd
from keras.models import Sequential, Model, Input
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Nadam
from sklearn.model_selection import train_test_split
from scipy.stats.stats import pearsonr
import time
from sklearn.preprocessing import MinMaxScaler
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Bidirectional
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold


np.random.seed(2020)


class DNN(object):
    def __init__(self):
        self.NB_EPOCH = 200
        self.BATCH_SIZE = 1000
        self.VERBOSE = 1
        self.OPTIMIZER = Adam()
        self.N_HIDDEN = 128
        self.VALIDATION_SPLIT = 0.14

    def normalize(self, X):
        X = np.array(X)
        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(X)
        X = scaler_X.transform(X)

        X = pd.DataFrame(X)

        self.scaler_X = scaler_X

        return X

    def split(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, shuffle=False)

        self.X_train = X_train
        self.y_train = Y_train
        self.X_test = X_test
        self.y_test = Y_test

    def pca(self, X, n=3):
        from sklearn.decomposition import PCA
        p = PCA(n_components=n)
        p.fit(X)
        X_new = p.transform(X)
        return X_new

    def knn(self):
        from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
        # nca = NeighborhoodComponentsAnalysis()
        model = KNeighborsClassifier(n_neighbors=2)
        # model = Pipeline([('nca', nca), ('knn', kn_model)])
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        print(acc)
        return acc

    def svc(self, greedy_search=False):
        from sklearn.svm import SVC

        degrees = np.arange(.5, 30, .5)
        coef0s  = np.arange(.5, 30, .5)
        tols = np.arange(.5, 30, .5)

        params = {'degree': degrees, 'coef0': coef0s, 'tol': tols}

        # model = SVC(kernel='poly', degree=2) # degree: 7,6,4,2
        # model = SVC(kernel='sigmoid')
        # model = SVC(kernel='poly', degree=6, coef0=3) #coef0: 3,5,6,7
        # model = SVC(kernel='poly', degree=6, coef0=7, tol=2) # tol:2
        model = SVC(kernel='poly', degree=6, coef0=7, tol=2)
        if greedy_search:
            model = GridSearchCV(estimator=model, param_grid=params, cv=KFold(random_state=42), verbose=10)

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(acc)
        print(report)

        # 87%

    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(solver='saga') # newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(acc)
        print(report)

        # 75%

    def bagging(self):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier



        model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=1000, bootstrap=True, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(acc)
        print(report)

        # 90%

    def random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(acc)
        print(report)

        # 90%

    def adaboost(self):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        model = AdaBoostClassifier(DecisionTreeClassifier())
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(acc)
        print(report)

    def stacking(self):
        from sklearn.ensemble import StackingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)),
            ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
        ]
        clf = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression()
        )

        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(acc)
        print(report)

    def decission_tree(self):
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(random_state=0, max_depth=10)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        cf = confusion_matrix(self.y_test, y_pred)
        print(cf)
        acc = accuracy_score(self.y_test, y_pred)
        print(acc)


    def dnn(self, path=None, name=None):
        with tf.device('/device:CPU:0'):
            model = Sequential()
            model.add(Dense(self.X_train.shape[1], input_shape=(self.X_train.shape[1],)))
            model.add(Activation('sigmoid'))
            for i in range(3):
                model.add(Dense(self.N_HIDDEN))
                model.add(Activation('sigmoid'))
            model.add(Dense(10))
            model.add(Activation('sigmoid'))
            model.add(Dense(1))
            model.add(Activation('softmax'))
            model.summary()

            model.compile(loss='binary_crossentropy',
                          optimizer=SGD(),
                          metrics=['accuracy'])
            history = model.fit(self.X_train, self.y_train,
                                epochs=1000,
                                validation_split=.1,
                                verbose=self.VERBOSE)

            y_pred = model.predict(self.X_test)

            y_pred = y_pred.reshape(-1)

            test_results = model.evaluate(self.X_test, self.y_test, verbose=1)
            print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1] * 100}%')

            # Plot decision boundary
            cf = confusion_matrix(self.y_test, y_pred)
            print(cf)

            # plot_decision_regions(self.X_test, self.y_test.flatten(), clf=model, legend=2)
            # plt.show()

            return

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

    def SVC(self):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.y_train = self.y_train.flatten()

        # Training a classifier
        svm = SVC(gamma='auto')
        svm.fit(self.X_train, self.y_train)

        # Plotting decision regions
        fig, axarr = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        values = [-4.0, -1.0, 1.0, 4.0]
        width = 0.75
        for value, ax in zip(values, axarr.flat):
            plot_decision_regions(self.X_train, self.y_train, clf=svm,
                                  filler_feature_values={2: value},
                                  filler_feature_ranges={2: width},
                                  legend=2, ax=ax)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('Feature 3 = {}'.format(value))

        # Adding axes annotations
        fig.suptitle('SVM on make_blobs')
        plt.tight_layout()
        plt.show()

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

