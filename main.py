import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


import os
print(os.getcwd())
os.nice(8)

from preprocessing import Preprocess
from regressor import Regressor



def split_data_to_models(data):
    budget_1k = data[(data['low_budget'] == True) & (data['budget'] < 7)]
    budget_10k = data[(data['low_budget'] == True) & (data['budget'] >= 7)]
    new_budget = data[(data['new_budget'] == True) & (data['no_budget'] == True)]
    no_budget = data[(data['new_budget'] == False) & (data['no_budget'] == True)]
    regular = data[(data['low_budget'] == False) & (data['no_budget'] == False)]
    return budget_1k, budget_10k, regular, new_budget, no_budget


def fit_predict_regressor(train, test, drop_cols, file_name):
    print(file_name)
    y_train, y_test = train['revenue'], test['revenue']
    X_train = train.drop(columns=['revenue'] + drop_cols)
    X_test = test.drop(columns=['revenue'] + drop_cols)
    reg = Regressor('xgboost', 50, 20)
    reg.fit(X_train, y_train)
    reg.save(file_name + '.pkl')
    features_scores = list(zip(X_train.columns.tolist(), reg.feature_importances_))
    for c, s in sorted(features_scores, key=lambda x: x[1], reverse=True)[:25]:
        print(c, s)
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print("MSE train: ", mse_train)
    print("MSE test: ", mse_test)
    with open('results_file.txt', 'a') as res:
        res.write(f'{file_name} - train: {np.round(mse_train, 3)} test: {np.round(mse_test, 3)}\n')
    print(file_name + '_drop')
    drop_reg = Regressor('xgboost', 50, 20)
    drop_cols_reg = [c for c, s in features_scores if s < 0.001]
    drop_reg.fit(X_train.drop(columns=drop_cols_reg), y_train)
    drop_reg.save(file_name + '_drop.pkl')
    features_scores = list(zip(X_train.columns.tolist(), reg.feature_importances_))
    for c, s in sorted(features_scores, key=lambda x: x[1], reverse=True)[:25]:
        print(c, s)
    y_train_pred = drop_reg.predict(X_train)
    y_test_pred = drop_reg.predict(X_test)
    drop_mse_train = mean_squared_error(y_train, y_train_pred)
    drop_mse_test = mean_squared_error(y_test, y_test_pred)
    print("MSE train: ", drop_mse_train)
    print("MSE test: ", drop_mse_test)
    with open('results_file.txt', 'a') as res:
        res.write(f'{file_name}_drop - train: {np.round(mse_train, 3)} test: {np.round(mse_test, 3)}\n')
    if mse_test < drop_mse_test:
        return reg
    else:
        return drop_reg


def fit_regressor_final(train, drop_cols, file_name):
    y_train = train['revenue']
    X_train = train.drop(columns=['revenue'] + drop_cols)
    interaction_columns = [c for c in train.columns if 'interaction' in c]
    ratio_columns = [c for c in train.columns if 'ratio' in c]
    best_score, best_reg = None, None
    for i, drop_this_columns in enumerate([[], interaction_columns, ratio_columns, interaction_columns + ratio_columns]):
        print(file_name)
        print(i, 'DROPING: ', drop_this_columns)
        reg = Regressor('xgboost', 50, 20)
        reg.fit(X_train.drop(columns=[c for c in X_train.columns if c in drop_this_columns]), y_train)
        reg.save(file_name + f'_{i}.pkl')
        features_scores = list(zip(reg.x_columns, reg.feature_importances_))
        for c, s in sorted(features_scores, key=lambda x: x[1], reverse=True)[:25]:
            print(c, s)
        y_train_pred = reg.predict(X_train.drop(columns=[c for c in X_train.columns if c in drop_this_columns]))
        mse_train = mean_squared_error(y_train, y_train_pred)
        reg_cv_score = -reg.best_score
        print("MSE CV: ", reg_cv_score)
        print("MSE train: ", mse_train)
        with open('results_file.txt', 'a') as res:
            res.write(f'{file_name} - train: {np.round(mse_train, 3)} CV: {np.round(reg_cv_score, 3)}\n')
        if best_score is None or reg_cv_score < best_score:
            best_score = reg_cv_score
            best_reg = reg
        print('dropping unimportant columns')
        drop_reg = Regressor('xgboost', 50, 20)
        drop_cols_reg = [c for c, s in features_scores if s < 0.001]
        drop_reg.fit(X_train.drop(columns=[c for c in X_train.columns if c in drop_this_columns] + drop_cols_reg), y_train)
        drop_reg.save(file_name + f'_{i}_drop.pkl')
        features_scores = list(zip(drop_reg.x_columns, reg.feature_importances_))
        for c, s in sorted(features_scores, key=lambda x: x[1], reverse=True)[:25]:
            print(c, s)
        y_train_pred = drop_reg.predict(X_train)
        drop_mse_train = mean_squared_error(y_train, y_train_pred)
        drop_reg_cv_score = -drop_reg.best_score
        print("MSE CV: ", drop_reg_cv_score)
        print("MSE train: ", drop_mse_train)
        with open('results_file.txt', 'a') as res:
            res.write(f'{file_name} - train: {np.round(mse_train, 3)} CV: {np.round(drop_reg_cv_score, 3)}\n')
        if best_score is None or reg_cv_score < best_score:
            best_score = drop_reg_cv_score
            best_reg = drop_reg
    best_reg.save(file_name + '_final.pkl')
    return best_reg


def predict(data, reg_regular, reg_new_budget, reg_no_budget):
    data['id'] = data.index
    budget_1k, budget_10k, regular, new_budget, no_budget = split_data_to_models(data)
    budget_1k['pred_revenue'] = budget_1k['budget'] * 1.4
    budget_10k['pred_revenue'] = budget_10k['budget'] * 1.1
    regular['pred_revenue'] = reg_regular.predict(regular.drop(columns=['id']))
    new_budget['pred_revenue'] = reg_new_budget.predict(new_budget.drop(columns=['id']))
    no_budget['pred_revenue'] = reg_no_budget.predict(no_budget.drop(columns=[c for c in no_budget.columns if 'budget' in c] + ['id']))
    predctions = pd.concat([budget_1k, budget_10k, regular, new_budget, no_budget], axis=0)[['id', 'pred_revenue']]
    return pd.merge(data, predctions, on='id', how='inner')


def predict_all(data, reg_regular, reg_new_budget, reg_no_budget, reg_no_budget_all):
    copied = data.copy()
    data['pred_revenue_1k'] = data['budget'] * 1.4
    data['pred_revenue_10k'] = data['budget'] * 1.1
    data['pred_revenue_regular'] = reg_regular.predict(copied.drop(columns=['revenue']))
    data['pred_revenue_new'] = reg_new_budget.predict(copied.drop(columns=['revenue']))
    data['pred_revenue_no_budget'] = reg_no_budget.predict(copied.drop(columns=[c for c in copied.columns if 'budget' in c] + ['revenue']))
    data['pred_revenue_no_budget_all'] = reg_no_budget_all.predict(copied.drop(columns=[c for c in copied.columns if 'budget' in c] + ['revenue']))
    return data


if __name__ == '__main__':
    train_path = 'train.tsv'
    test_path = 'test.tsv'
    train = pd.read_csv(train_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")
    print(train.shape, test.shape)
    preprocess = Preprocess(scale=False)
    print('preprocessing')
    train = pd.concat([train, test], axis=0).reset_index(drop=True)
    train = preprocess.fit_transform(train)
    preprocess.save(f'preprocess_final.pkl')
    print('finished preprocessing')
    y_train, y_test = train['revenue'], np.log(test['revenue'] + 1)
    test = preprocess.transform(test)
    test['revenue'] = y_test
    budget_1k_train, budget_10k_train, regular_train, new_budget_train, no_budget_train = split_data_to_models(train)
    reg_regular = fit_regressor_final(regular_train, [], file_name=f'regular')
    all_train = pd.concat([regular_train, new_budget_train, no_budget_train], axis=0)
    missing_budget_train = pd.concat([new_budget_train, no_budget_train], axis=0)
    reg_no_budget_all = fit_regressor_final(all_train, [c for c in all_train.columns if 'budget' in c], file_name=f'no_budget_columns_all')
    reg_no_budget_missing = fit_regressor_final(missing_budget_train, [c for c in missing_budget_train.columns if 'budget' in c], file_name=f'no_budget_columns_missing')
    if reg_no_budget_all.best_score > reg_no_budget_missing.best_score:
        reg_no_budget = reg_no_budget_all
    else:
        reg_no_budget = reg_no_budget_missing
    reg_no_budget.save('no_budget_final.pkl')
    reg_new_budget = fit_regressor_final(new_budget_train, [], file_name=f'new_budget')
    test_predicted = predict(test.drop(columns=['revenue']), reg_regular, reg_new_budget, reg_no_budget)
    print('Combined:', mean_squared_error(test['revenue'], test_predicted['pred_revenue']))