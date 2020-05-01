import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import os
print(os.getcwd())

from preprocessing import Preprocess
from preprocessings2 import Preprocess as Preprocess_e
from regressor import Regressor
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    train_path = 'train.tsv'
    test_path = 'test.tsv'
    for x_scale in [True, False]:
        for y_scale in [False, True]:
            for embed in [True, False]:
                if y_scale and embed:
                    continue
                file_prefix = f'x_{x_scale}_y_{y_scale}_embed_{embed}'
                with open('results_file.txt', 'a') as res:
                    res.write(f'Starting x_scale: {x_scale} - y_scale: {y_scale} - embedding: {embed}' + '\n')

                train = pd.read_csv(train_path, sep="\t")
                test = pd.read_csv(test_path, sep="\t")
                if embed:
                    preprocess = Preprocess_e(scale=x_scale)
                else:
                    preprocess = Preprocess(scale=x_scale)
                train = preprocess.fit_transform(train)
                y_train, y_test = train['revenue'], np.log(test['revenue'] + 1)
                preprocess.save(f'{file_prefix}_preprocess.pkl')
                train = train.drop(columns='revenue')
                test = preprocess.transform(test)
                if y_scale:
                    y_scaler = StandardScaler()
                    y_scaler.fit(y_train.values.reshape(-1, 1))
                    y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1))
                for regressor_str in ['xgboost', 'rfr']:
                    reg = Regressor(regressor_str, 35, 20)
                    if y_scale:
                        reg.fit(train, y_train_scaled.ravel())
                    else:
                        reg.fit(train, y_train)
                    reg.save(f'{file_prefix}_model_{regressor_str}.pkl')
                    print(f'{file_prefix}_model_{regressor_str}')
                    features_scores = list(zip(train.columns.tolist(), reg.feature_importances_))
                    for c, s in sorted(features_scores, key=lambda x: x[1], reverse=True)[:25]:
                        print(c, s)
                    y_train_pred = reg.predict(train)
                    y_test_pred = reg.predict(test)
                    if y_scale:
                        y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
                        y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
                    mse_train = mean_squared_error(y_train, y_train_pred)
                    mse_test = mean_squared_error(y_test, y_test_pred)
                    print("MSE train: ", mse_train)
                    print("MSE test: ", mse_test)
                    with open('results_file.txt', 'a') as res:
                        res.write(f'model: {regressor_str} - train: {np.round(mse_train, 3)} test: {np.round(mse_test, 3)}' + '\n')

                    drop_cols = [c for c, s in features_scores if s < 0.003]
                    reg = Regressor(regressor_str, 35, 20)
                    if y_scale:
                        reg.fit(train.drop(columns=drop_cols), y_train_scaled.ravel())
                    else:
                        reg.fit(train.drop(columns=drop_cols), y_train)
                    reg.save(f'{file_prefix}_model_{regressor_str}_drop.pkl')
                    print(f'{file_prefix}_model_{regressor_str}_drop')
                    features_scores = list(zip(train.drop(columns=drop_cols).columns.tolist(), reg.feature_importances_))
                    for c, s in sorted(features_scores, key=lambda x: x[1], reverse=True)[:25]:
                        print(c, s)
                    y_train_pred = reg.predict(train.drop(columns=drop_cols))
                    y_test_pred = reg.predict(test.drop(columns=drop_cols))
                    if y_scale:
                        y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))
                        y_test_pred = y_scaler.inverse_transform(y_test_pred.reshape(-1, 1))
                    mse_train = mean_squared_error(y_train, y_train_pred)
                    mse_test = mean_squared_error(y_test, y_test_pred)
                    print("MSE train: ", mse_train)
                    print("MSE test: ", mse_test)
                    with open('results_file.txt', 'a') as res:
                        res.write(f'model: {regressor_str} DROP: True - train: {np.round(mse_train, 3)} test: {np.round(mse_test, 3)}' + '\n')
