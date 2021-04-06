import talib
import pandas as pd
from DNN import DNN
import matplotlib.pyplot as plt


def main():
    # Combining Indicators
    # Index
    djia = pd.read_excel('./DJIA.xls')
    djia = djia.dropna()
    djia = djia.iloc[:, 1]

    indicators = [
        talib.RSI,
        talib.EMA
    ]

    net = DNN(10, None)
    X, Y = net.prepare(djia)
    X, Y = net.normalize(X, Y)

    for i in indicators:
        tmp = i(djia)
        tmp, _ = net.prepare(tmp)
        tmp, _ = net.normalize(tmp, _)
        X = pd.concat([X, tmp], axis=1)

    new_X = pd.DataFrame()
    new_Y = list()
    for index, row in X.iterrows():
        is_nan = row.isna()
        if not is_nan.any():
            new_X = pd.concat([new_X, row], axis=1)
            new_Y.append(Y[index][0])

    new_X = X.T

    # net.split(new_X, new_Y)

    # _ = net.dnn()
    # print(_)


def main2():
    # One by One

    min_index = 100
    max_index = 1100

    # Get Function Names
    indicators = talib.__TA_FUNCTION_NAMES__

    djia = pd.read_excel('./DJIA.xls')
    djia = djia.dropna()
    djia = djia.iloc[:, 1]

    net = DNN(5, None)
    X, Y = net.prepare(djia)
    X, Y = net.normalize(X, Y)
    # X = X.iloc[min_index:max_index, :]
    Y = Y[min_index:max_index]

    # net.split(X, Y)
    # rmse, mse, mae, mape = net.dnn(name='original')
    # print(rmse, mse, mae, mape)

    # results = {'name': list(), 'rmse': list(), 'mse': list(), 'mae': list(), 'mape': list()}
    #
    # for i in indicators:
    #     try:
    #         tmp = eval(f'talib.{i}')
    #         tmp = tmp(djia)
    #         # raw = tmp.copy()
    #         tmp, _ = net.prepare(tmp)
    #         tmp, _ = net.normalize(tmp, _)
    #     except:
    #         continue
    #
    #     data = pd.concat([X, tmp], axis=1)
    #     data = data.iloc[min_index:max_index, :]
    #     if data.isnull().values.any():
    #         continue
    #
    #     # plt.plot(raw[min_index:max_index])
    #     # plt.savefig(f'./indicators/{i}.png')
    #     # plt.clf()
    #
    #     net.split(data, Y)
    #
    #     rmse, mse, mae, mape = net.dnn(name=i)
    #     print(rmse, mse, mae, mape)
    #
    #     results['name'].append(i)
    #     results['rmse'].append(rmse)
    #     results['mse'].append(mse)
    #     results['mae'].append(mae)
    #     results['mape'].append(mape)
    #
    # results_dataframe = pd.DataFrame(results)
    # results_dataframe.to_excel('results.xlsx')


def main3():
    # One by One

    min_index = 100
    max_index = 1100

    # Get Function Names
    indicators = talib.__TA_FUNCTION_NAMES__

    djia = pd.read_excel('./DJIA.xls')
    djia = djia.dropna()
    djia = djia.iloc[:, 1]

    net = DNN(5, None)
    X, Y = net.prepare(djia)
    X, Y = net.normalize(X, Y)
    X = X.iloc[min_index:max_index, :]
    Y = Y[min_index:max_index]

    # net.split(X, Y)
    # rmse, mse, mae, mape = net.dnn(name='original')
    # print(rmse, mse, mae, mape)

    results = {'name': list(), 'rmse': list(), 'mse': list(), 'mae': list(), 'mape': list()}

    net = DNN(2, None)

    for i in indicators:
        try:
            tmp = eval(f'talib.{i}')
            tmp = tmp(djia)
            tmp, _ = net.prepare(tmp)
            tmp, _ = net.normalize(tmp, _)
        except:
            continue
        tmp = tmp.iloc[min_index:max_index, :]
        if tmp.isnull().values.any():
            continue

        X = pd.concat([X, tmp], axis=1)


    net.split(X, Y)

    rmse, mse, mae, mape = net.dnn(name=i)
    print(rmse, mse, mae, mape)

    results['name'].append(i)
    results['rmse'].append(rmse)
    results['mse'].append(mse)
    results['mae'].append(mae)
    results['mape'].append(mape)

    results_dataframe = pd.DataFrame(results)
    results_dataframe.to_excel('results.xlsx')


def main4():
    # One by One LSTM

    min_index = 100
    max_index = 1100

    # Get Function Names
    indicators = talib.__TA_FUNCTION_NAMES__

    djia = pd.read_excel('./DJIA.xls')
    djia = djia.dropna()
    djia = djia.iloc[:, 1]

    net = DNN(5, None)
    X, Y = net.prepare(djia)
    X, Y = net.normalize(X, Y)
    # X = X.iloc[min_index:max_index, :]
    Y = Y[min_index:max_index]

    # net.split(X, Y)
    # rmse, mse, mae, mape = net.lstm(name='original')
    # print(rmse, mse, mae, mape)

    results = {'name': list(), 'rmse': list(), 'mse': list(), 'mae': list(), 'mape': list()}

    for i in indicators:
        try:
            tmp = eval(f'talib.{i}')
            tmp = tmp(djia)
            # raw = tmp.copy()
            tmp, _ = net.prepare(tmp)
            tmp, _ = net.normalize(tmp, _)
        except:
            continue

        data = pd.concat([X, tmp], axis=1)
        data = data.iloc[min_index:max_index, :]
        if data.isnull().values.any():
            continue

        # plt.plot(raw[min_index:max_index])
        # plt.savefig(f'./indicators/{i}.png')
        # plt.clf()

        net.split(data, Y)

        rmse, mse, mae, mape = net.lstm(name=i)
        print(rmse, mse, mae, mape)

        results['name'].append(i)
        results['rmse'].append(rmse)
        results['mse'].append(mse)
        results['mae'].append(mae)
        results['mape'].append(mape)

    results_dataframe = pd.DataFrame(results)
    results_dataframe.to_excel('results.xlsx')



def main5():
    # Sequence LSTM

    min_index = 100
    max_index = 1100

    # Get Function Names
    indicators = talib.__TA_FUNCTION_NAMES__

    djia = pd.read_excel('./DJIA.xls')
    djia = djia.dropna()
    djia = djia.iloc[:, 1]

    net = DNN(8, None)
    X, Y = net.prepare(djia)
    X, Y = net.normalize(X, Y)
    X = X.iloc[min_index:max_index, :]
    Y = Y[min_index:max_index]

    net.split(X, Y)
    rmse, mse, mae, mape = net.lstm_with_sequence(name='original')
    print(rmse, mse, mae, mape)

    # results = {'name': list(), 'rmse': list(), 'mse': list(), 'mae': list(), 'mape': list()}
    #
    # for i in indicators:
    #     try:
    #         tmp = eval(f'talib.{i}')
    #         tmp = tmp(djia)
    #         # raw = tmp.copy()
    #         tmp, _ = net.prepare(tmp)
    #         tmp, _ = net.normalize(tmp, _)
    #     except:
    #         continue
    #
    #     data = pd.concat([X, tmp], axis=1)
    #     data = data.iloc[min_index:max_index, :]
    #     if data.isnull().values.any():
    #         continue
    #
    #     # plt.plot(raw[min_index:max_index])
    #     # plt.savefig(f'./indicators/{i}.png')
    #     # plt.clf()
    #
    #     net.split(data, Y)
    #
    #     rmse, mse, mae, mape = net.lstm_with_sequence(name=i)
    #     print(rmse, mse, mae, mape)
    #
    #     results['name'].append(i)
    #     results['rmse'].append(rmse)
    #     results['mse'].append(mse)
    #     results['mae'].append(mae)
    #     results['mape'].append(mape)
    #
    # results_dataframe = pd.DataFrame(results)
    # results_dataframe.to_excel('results.xlsx')


def main6():
    # Sequence LSTM with corr

    min_index = 100
    max_index = 1100

    # Get Function Names
    indicators = talib.__TA_FUNCTION_NAMES__

    data = pd.read_excel('./DJIA.xls')
    data = data.dropna()
    data = data.iloc[:, 1]
    raw_data = data.copy()

    net = DNN(64, None)
    X, Y = net.prepare(data)
    X, Y = net.normalize(X, Y)
    X = X.iloc[min_index:max_index, :]
    Y = Y[min_index:max_index]
    tmp_X = X.copy()

    net.split(X, Y)
    rmse, mse, mae, mape = net.CNN_LSTM(name='original')
    print(rmse, mse, mae, mape)

    # results = {'name': list(), 'rmse': list(), 'mse': list(), 'mae': list(), 'mape': list()}
    #
    # for i in indicators:
    #     try:
    #         tmp = eval(f'talib.{i}')
    #         tmp = tmp(raw_data)
    #         raw = tmp.copy()
    #         tmp, _ = net.prepare(tmp)
    #         tmp, _ = net.normalize(tmp, _)
    #     except:
    #         continue
    #
    #     test = pd.concat([tmp_X, tmp], axis=1)
    #     test = test.iloc[min_index:max_index, :]
    #     if test.isnull().values.any():
    #         continue
    #     test = pd.concat([raw_data, raw], axis=1)
    #     test = test.iloc[min_index:max_index, :]
    #     c = test.corr(method='pearson')
    #     if c.iloc[0, 1] < .9:
    #         continue
    #     X = pd.concat([X, tmp], axis=1)
    #
    # X = X.iloc[min_index:max_index, :]
    #
    # net.split(X, Y)
    #
    # rmse, mse, mae, mape = net.lstm_with_sequence(name='all')
    # print(rmse, mse, mae, mape)
    #
    # results['name'].append(i)
    # results['rmse'].append(rmse)
    # results['mse'].append(mse)
    # results['mae'].append(mae)
    # results['mape'].append(mape)
    #
    # results_dataframe = pd.DataFrame(results)
    # results_dataframe.to_excel('results.xlsx')


def get_corr():
    min_index = 100
    max_index = 1100

    # Get Function Names
    indicators = talib.__TA_FUNCTION_NAMES__

    data = pd.read_excel('./DJIA.xls')
    data = data.dropna()
    data = data.iloc[:, 1]
    raw_data = data.copy()

    for i in indicators:
        try:
            tmp = eval(f'talib.{i}')
            tmp = tmp(raw_data)
        except:
            continue
        try:
            test = pd.concat([raw_data, tmp], axis=1)
        except:
            continue
        test = test.iloc[min_index:max_index, :]
        if test.isnull().values.any():
            continue
        data = pd.concat([data, tmp], axis=1)
    data = data.iloc[min_index:max_index, :]
    data.to_excel('all_data.xlsx')
    data2 = data.corr(method='pearson')
    data2.to_excel('all_corr.xlsx')


if __name__ == '__main__':
    main6()
