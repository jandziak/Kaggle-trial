import os
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
print(os.getcwd())
path = 'C:\\Users\\iWindows\\Desktop\\Python Prudential'
os.chdir(path)


def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)


def get_params():
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.05
    params["min_child_weight"] = 160
    params["subsample"] = 1
    params["colsample_bytree"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 5
    plst = list(params.items())
    return plst


def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int) == sv] = data[0, data[0].astype(int) == sv] + bin_offset
    score = scorer(data[1], data[2])
    return score


def load_data(train_path, test_path):
    print("Load the data using pandas")
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    return train_set, test_set



def transform_data(train_set, test_set):
    # combine train and test
    all_data = train_set.append(test_set)

    # Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
    # create any new variables
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

    # factorize categorical variables
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

    print('Eliminate missing values')
    # Use -1 for any others
    all_data.fillna(-1, inplace=True)

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)

    # split train and test
    train_set = all_data[all_data['Response'] > 0].copy()
    test_set = all_data[all_data['Response'] < 1].copy()
    return train_set, test_set


def fit_xgb_model(train, test, eta):
    # global variables
    columns_to_drop = ['Id', 'Response', 'Medical_History_1']
    xgb_num_rounds = 20
    eta_list = eta  # [0.05]* 200
    eta_list = eta_list + [0.02] * 1000
    # convert data to xgb data structure
    xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)

    # get the parameters for xgboost
    plst = get_params()
    print(plst)

    # train model
    model = xgb.train(plst, xgtrain, xgb_num_rounds, learning_rates=eta_list)

    return model, xgtrain




def train_offset(model, xg_train_set, train):
    # get preds
    train_preds = model.predict(xg_train_set, ntree_limit=model.best_iteration)
    print('Train score is:', eval_wrapper(train_preds, train['Response']))
    train_preds = np.clip(train_preds, -0.99, 8.99)
    num_classes = 8
    # train offsets
    offsets_vec = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
    data = np.vstack((train_preds, train_preds, train['Response'].values))
    for j in range(num_classes):
        data[1, data[0].astype(int) == j] = data[0, data[0].astype(int) == j] + offsets_vec[j]
    for j in range(num_classes):
        train_offset_vec = lambda x: -apply_offset(data, x, j)
        offsets_vec[j] = fmin_powell(train_offset_vec, offsets_vec[j])
    return offsets_vec




def make_submission(test_set, model, offsets):
    # apply offsets to test
    columns_to_drop = ['Id', 'Response', 'Medical_History_1']
    num_classes = 8
    xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)
    test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    test_preds = np.clip(test_preds, -0.99, 8.99)
    data = np.vstack((test_preds, test_preds, test['Response'].values))
    for j in range(num_classes):
        data[1, data[0].astype(int) == j] = data[0, data[0].astype(int) == j] + offsets[j]

    final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

    preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
    preds_out = preds_out.set_index('Id')
    preds_out.to_csv('xgb_offset_submission.csv')


train, test = load_data('./input/train.csv', './input/test.csv')
train, test = transform_data(train, test)
model, xgtrain = fit_xgb_model(train, test, [0.5*200])
make_submission(test, model, offsets)offsets = train_offset(model, xgtrain, train)