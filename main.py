import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from preprocessing import Preprocess
from regressor import Regressor


if __name__ == '__main__':
    # train_path = 'train.tsv'
    test_path = 'test.tsv'
    # train = pd.read_csv(train_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")
    # preprocess = Preprocess()
    # train = preprocess.fit_transform(train)
    # preprocess.save('preprocess2.pkl')
    #
    # reg = Regressor('regressor', 50, 50)
    # reg.fit(train.drop(columns='revenue'), train['revenue'])
    # reg.save('fiited_model2.pkl')

    preprocess2 = Preprocess.load('preprocess2.pkl')
    test = preprocess2.transform(test)
    reg = Regressor.load('fiited_model2.pkl')
    for c,s in zip(test.drop(columns=['revenue']).columns.tolist(), reg.feature_importances_):
        print(c, s)
    y_pred = reg.predict(test.drop(columns='revenue'))
    y_test = np.log(test['revenue'] + 1)
    print("result: ", mean_squared_error(y_pred, y_test))