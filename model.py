import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import xgboost as xg
from sklearn.metrics import classification_report
import sys, json, jsonpickle
from pandas.io.json import json_normalize
import json
from sklearn import metrics
from scipy.sparse import csc_matrix
import random
import datetime
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix


def preprocessing(dataset, x_axis, y_axis, model_name, model_type):

    print(type(dataset))
    df = pd.DataFrame.from_records(dataset)
    process_df = dataset_preprocessing(df)
    print(process_df[(x_axis)])
    if model_type == 'linear_reg':

        lr, coeff, xtest = linear_model_creation(process_df[(x_axis)],
                                                 process_df[(y_axis)],
                                                 model_name)
        print('model created')

        return process_df, lr, coeff, xtest

    elif model_type == 'log_regression':
        modal_whole, modal_score, c_matrix, x_delete = log_modal_creation(
            process_df[(x_axis)], process_df[(y_axis)], model_name)
        return modal_whole, modal_score, c_matrix, x_delete

    # elif model_type == 'xgboost':
    #     print('xgboost model')
    #     xg_boost, coeff, xtest = xgboost_model_creation(
    #         process_df[x_axis], process_df[y_axis], model_name)
    #     print('model created')
    #     return process_df, xg_boost, coeff, xtest


#preprocess columns data and prepare new dataframe from
def dataset_preprocessing(dataset):
    try:

        dataset.dropna(how='all', inplace=True)
        dataset.fillna(method='ffill', inplace=True)

        return dataset

    except Exception as e:
        return e


#linear Regression Model
def linear_model_creation(x, y, model_name):

    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    coeff = model_details(model, X_test, y_test)
    print(X_test)
    X_test['y_test'] = y_test

    model = jsonpickle.encode(model)
    coeff = pd.Series(coeff).to_json(orient='values')
    return model, coeff, X_test

    #XGBoost Model

    # def xgboost_model_creation(x, y, model_name):

    #     model = xg.XGBRegressor(n_estimators=100,
    #                         learning_rate=0.08,
    #                         gamma=0,
    #                         subsample=0.75,
    #                         colsample_bytree=1,
    #                         max_depth=7)
    #     X_train, X_test, y_train, y_test = train_test_split(x,
    #                                                          y,
    #                                                     test_size=0.3,
    #                                                     random_state=101)

    # model.fit(X_train.values, y_train.values)
    # X_test['y_test'] = y_test.values

    # coeff = []
    # model = jsonpickle.encode(model)

    # return model, coeff, X_test


def model_pred_auto(model, x_test, y_axis, model_type):

    model_pr = jsonpickle.decode(model)
    x_test = pd.read_json(x_test, orient='records')
    y_test = x_test['y_test']
    test_auto = x_test.drop(['y_test'], axis=1)
    df_coeflen = len(test_auto.columns)
    print(type(test_auto))
    print(type(y_test))

    if model_type == "linear_reg":

        pred = model_pr.predict(test_auto)
        print(pred)
        print(model_pr)
        # print(pred)
        print("coeff:", model_pr.coef_)
        print("Intercept:", model_pr.intercept_)
        r_square = r2_score(y_test, pred)
        print("r_square", r_square)

    elif model_type == "xgboost":

        pred = model_pr.predict(test_auto.values)

        print("xgboost auto:", pred)
        r_square = r2_score(y_test, pred)
        print("r_square", r_square)

    r_square = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    adj = 1 - float(len(y_test) - 1) / (len(y_test) - df_coeflen - 1) * (
        1 - metrics.r2_score(y_test, pred))

    return pred, r_square, adj, mse


def model_details(model, X_test, y_test):

    print("coeff len:", len(model.coef_))
    coeff = [model.coef_, model.intercept_]

    return coeff


def y_testFunction(df, model):
    y_test = df.reindex(np.random.permutation(df.index))
    pred = model.predict(y_test.values)
    y_test = pd.Series(pred)
    # print("y_test predition:",y_test)
    # print("y_test preditin datatype",type(y_test))
    return y_test


def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()


def model_pred_manual(model, x_axis, predict_data):
    model_pr = jsonpickle.decode(model)
    print("Model:", model_pr)
    x_test = pd.DataFrame.from_records(predict_data)
    # print(x_test)
    # column_last = list(x_test.columns.values)
    # column_last = column_last[2]
    # print("column_last:",column_last)
    # week = pd.DataFrame(x_test['Week'])
    # x_test_week = datetime.datetime(week)
    # x_test['Week'].apply(lambda x: pd.to_datetime(x,errors='coerce', format = "%Y-%m-%dT%H:%M:%S"))
    # ss = week.to_json()
    # print("jsondate",ss)
    # print(week)
    # print(type(week))
    # date = datetime.datetime.strptime(week, "%Y-%m-%dT%H:%M:%S")
    # timestamp = pd.to_datetime(week[1:]).strftime("%Y-%m-%dT%H:%M:%S")
    # print(x_test)
    # print(x_test_week)
    # print("X_test columns",x_test.columns)

    # x_test['week'] = week

    # x_test['week'] = week
    # x_test = x_test.dropna(how='any',axis=0)

    # x_test.dropna(axis=0,inplace=True)
    x_test = x_test.dropna(axis=0, how='any', thresh=None, subset=None)
    abc = x_test
    xtest_last = pd.DataFrame()
    # print("ater dropping:",x_test)
    xtest_last = x_test['Week']

    print("xtest_procesing:", x_test)

    x_test = x_test[x_axis]

    y_test = x_test.reindex(index=x_test.index[::-1])

    y_test = y_testFunction(x_test, model_pr)
    # print("ytest:",y_test)
    pred = model_pr.predict(x_test)
    # print("Main prediction:",y_test)
    df_coeflen = len(x_test.columns)
    df_dt = pd.DataFrame(pred)
    # print("After prediction:",xtest_last)

    df_dt['Week'] = xtest_last
    df_dt.columns = ['prediction', 'Week']
    df_dt.dropna(inplace=True)
    # print("df_dt",df_dt)
    r_square = 0
    mse = 0
    adj = 0
    # print("len of df_dt",len(df_dt['Week']))
    r_square = r2_score(y_test, pred)
    # print("r_Square:",r_square)
    mse = mean_squared_error(y_test, pred)
    # print(mse)
    adj = 1 - float(len(y_test) - 1) / (len(y_test) - df_coeflen - 1) * (
        1 - metrics.r2_score(y_test, pred))

    return df_dt, x_test, r_square, adj, mse


def log_modal_creation(x, y, model_name):
    data = x
    print(x)
    data_types = data.dtypes.to_dict()
    for x in data_types.keys():
        if(data_types[x] == "object"):
            temp = pd.get_dummies(data[x],drop_first=True)
            data = pd.concat([data,temp],axis=1)
            del data[x]
            print(x)
    
    X = data
    Y = y
    
    mod = sm.OLS(Y,X)
    fii = mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|'].to_dict()
    temp_keys = p_values.keys()
    print(fii.summary2())
    x_delete = []
    for x in temp_keys:
        if(p_values[x] > 0.05 or p_values[x] < -0.05):
            del data[x]
            print(x)
            x_delete.append(x)


    X = data
    Y=y

    mod = sm.OLS(Y, X)
    fii = mod.fit()
    p_values = fii.summary2()
    t_values = fii.summary2().tables[1]['t'].to_dict()
    dictonary = {}
    keep_var = []

    if(len(t_values) > 20):
        for x in t_values:
            dictonary[x] = abs(t_values[x])

        dictonary = sorted(dictonary.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        
        print(dictonary)
        for x in range(0, 20):
            keep_var.append(dictonary[x][0])
        X = data[keep_var]
    else:
        keep_var = X.columns
        print(keep_var)

    temp = list(set(keep_var))
    X = data[temp]
    print(len(X.columns))
    print(X.columns)
    mod = sm.OLS(Y, X)
    fii = mod.fit()

    keep_var = X.columns
    print(keep_var)
    p_t_value = list(fii.summary2().tables[1]['P>|t|'])
    std_err = list(fii.summary2().tables[1]['Std.Err.'])
    coef = list(fii.summary2().tables[1]['Coef.'])
    t_value = list(fii.summary2().tables[1]['t'])
    print(len(keep_var))
    print(len(p_t_value))
    temp_create = [keep_var, p_t_value, std_err, coef, t_value]
    data_frame_list = pd.DataFrame(temp_create)
    keep_var = data_frame_list.to_json(orient='values')


    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
    modal = LogisticRegression()
    modal.fit(x_train,y_train)
    modal_score = modal.score(x_test,y_test)
    y_pred = modal.predict(x_train)
    print(y_pred)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(modal.score(x_test, y_test)))
    c_matrix = confusion_matrix(y_train,y_pred)

    c_matrix = c_matrix.tolist()
    print(c_matrix)
    modal_whole = jsonpickle.encode(modal)

    return modal_whole,modal_score,c_matrix,keep_var


def log_pred(model, x_axis, keep, dataset):
    modal = jsonpickle.decode(model)
    df = pd.DataFrame.from_records(dataset)
    df = df.dropna()
    process_df = dataset_preprocessing(df)
    print(x_axis)
    data = process_df[(x_axis)]
    data_types = data.dtypes.to_dict()
    for x in data_types.keys():
        if(data_types[x] == "object"):
            temp = pd.get_dummies(data[x], drop_first=True)
            data = pd.concat([data, temp], axis=1)
            del data[x]
            print(x)
    temp_x = json.loads(keep)
    temp_x_cols= temp_x[0]
    print(len(temp_x_cols))
    print(data[temp_x_cols].columns)
    # predict_res = modal.predict(data[temp_x_cols])
    predict_res = modal.predict_proba(data[temp_x_cols])[:,1]
    print(predict_res)
    predict_res = list(predict_res)
    return predict_res
