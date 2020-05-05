import argparse
import numpy as np
import pandas as pd
from regressor import Regressor
from preprocessing import Preprocess

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

#####

preprocess_file = 'preprocess_final.pkl'
reg_regular_file = 'regular_final.pkl'
reg_no_budget_all_file = 'no_budget_all_final.pkl'
reg_no_budget_file = 'no_budget_final.pkl'
reg_new_budget_file = 'new_budget_final.pkl'


def split_data_to_models(data):
    budget_1k = data[(data['low_budget'] == True) & (data['budget'] < 7)]
    budget_10k = data[(data['low_budget'] == True) & (data['budget'] >= 7)]
    new_budget = data[(data['new_budget'] == True) & (data['no_budget'] == True)]
    no_budget = data[(data['new_budget'] == False) & (data['no_budget'] == True)]
    regular = data[(data['low_budget'] == False) & (data['no_budget'] == False)]
    return budget_1k, budget_10k, regular, new_budget, no_budget


def predict(data, reg_regular, reg_new_budget, reg_no_budget):
    data = data.reset_index()
    budget_1k, budget_10k, regular, new_budget, no_budget = split_data_to_models(data.copy())
    budget_1k['pred_revenue'] = budget_1k['budget'] * 1.4
    budget_10k['pred_revenue'] = budget_10k['budget'] * 1.1
    regular['pred_revenue'] = reg_regular.predict(regular.drop(columns=['id']))
    new_budget['pred_revenue'] = reg_new_budget.predict(new_budget.drop(columns=['id']))
    no_budget['pred_revenue'] = reg_no_budget.predict(no_budget.drop(columns=[c for c in no_budget.columns if 'budget' in c] + ['id']))
    predictions = pd.merge(data[['id']], pd.concat([budget_1k, budget_10k, regular, new_budget, no_budget], axis=0)[['id', 'pred_revenue']]\
        .rename(columns={'pred_revenue':'revenue'}), on='id')
    predictions['revenue'] = predictions['revenue'].where(predictions['revenue'] > 1, predictions['revenue'].mean())
    predictions['revenue'] = np.exp(predictions['revenue']) - 1
    return predictions

preprocess = Preprocess.load(preprocess_file)
data = preprocess.transform(data.reset_index(drop=True))
reg_regular = Regressor.load(reg_regular_file)
reg_no_budget = Regressor.load(reg_no_budget_file)
reg_new_budget = Regressor.load(reg_new_budget_file)


prediction_df = predict(data, reg_regular, reg_new_budget, reg_no_budget)
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Utility function to calculate RMSLE
# def rmsle(y_true, y_pred):
#     """
#     Calculates Root Mean Squared Logarithmic Error between two input vectors
#     :param y_true: 1-d array, ground truth vector
#     :param y_pred: 1-d array, prediction vector
#     :return: float, RMSLE score between two input vectors
#     """
#     assert y_true.shape == y_pred.shape, \
#         ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
#     return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))
#
#
# ### Example - Calculating RMSLE
# res = rmsle(data['revenue'], prediction_df['revenue'])
# print("RMSLE is: {:.6f}".format(res))


