import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import os
print(os.getcwd())


from preprocessings2 import Preprocess
from regressor import Regressor
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    scale = False
    y_scaler = None
    train_path = 'train.tsv'
    test_path = 'test.tsv'
    train = pd.read_csv(train_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")
    preprocess = Preprocess(scale=True)
    train = preprocess.fit_transform(train)
    preprocess.save('preprocess3.pkl')

    reg = Regressor('regressor', 50, 50)
    reg.fit(train.drop(columns='revenue'), train['revenue'])
    reg.save('fiited_model3.pkl')

    if scale:  # if we want to scale the revenue
        y_scaler = StandardScaler()
        y_scaler.fit(train['revenue'])
        train['revenue'] = y_scaler.transform(train['revenue'])

    preprocess2 = Preprocess.load('preprocess3.pkl')
    test = preprocess2.transform(test)
    reg = Regressor.load('fiited_model3.pkl')
    for c,s in zip(test.drop(columns=['revenue']).columns.tolist(), reg.feature_importances_):
        print(c, s)
    y_pred = reg.predict(test.drop(columns='revenue'))
    if scale:  # if we want to scale the revenue
        y_pred = y_scaler.inverse_transform(y_pred)

    y_test = np.log(test['revenue'] + 1)
    print("result: ", mean_squared_error(y_pred, y_test))