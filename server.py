from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pickle
import json
import model as m
import simplejson
import jsonpickle
import pandas as pd
import forcast as fc
import json
#import Bio as b
app = Flask(__name__)


@app.route('/modelcreate', methods=['POST'])
def get_model():

    req_data = request.get_json()

    dataset_json = req_data['Dataset']
    x_axis = req_data['x_axis']
    y_axis = req_data['y_axis']
    model_name = req_data['Model_name']
    model_type = req_data['Model_type']
    date = req_data['date']
    time = req_data['time']
    no_var = req_data['no_var']

    if(model_type == "log_regression"):
        modal_whole, modal_score, c_matrix, keep = m.preprocessing(
            dataset_json, x_axis, y_axis, model_name, model_type)
        #c_matrix = pd.Series(c_matrix).to_json()
        c_matrix = json.dumps(c_matrix)
        return jsonify({
            "Model_name": model_name,
            "Model_type": model_type,
            "Model": modal_whole,
            "Accuracy": modal_score,
            "c_matrix": c_matrix,
            "Keep": keep,
            "date": date,
            "time": time,
            "no_var": no_var,
            "x_axis": x_axis,
            "y_axis": y_axis
        })
    else:
        model_pre, model, coeff, xtest = m.preprocessing(
            dataset_json, x_axis, y_axis, model_name, model_type)
        coeff = pd.Series(coeff).to_json(orient='values')
        df_js = model_pre.to_json(orient='records')
        xtest = xtest.to_json(orient='records')
        print("coeff:", coeff)
        print("np_of_var : "+no_var)
        return jsonify({
            "Model_name": model_name,
            "Coefficient and Intercept": coeff,
            "Model_type": model_type,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "Model": model,
            "Dataset": df_js,
            "X_test": xtest,
            "date": date,
            "time": time,
            "no_var": no_var
        })


@app.route("/predict", methods=['POST'])
def predict_auto():

    req_data = request.get_json()
    if(req_data['Model_type'] == "log_regression"):
        return jsonify(req_data)
    else:
        dataset = req_data['Dataset']
        x_axis = req_data['x_axis']
        y_axis = req_data['y_axis']
        model_name = req_data['Model_name']
        model_type = req_data['Model_type']
        model = req_data['Model']
        x_test = req_data['X_test']
        time = req_data['time']
        date = req_data['date']
        no_var = req_data['no_var']
        pred, r_square, adj, mse = m.model_pred_auto(model, x_test, y_axis,
                                                     model_type)
        pred = pd.Series(pred).to_json(orient='values')
        coff = req_data['Coefficient and Intercept']
        return jsonify({
            "Model name": model_name,
            "Model_type": model_type,
            "Model": model,
            "X_test": x_test,
            "X axis": x_axis,
            "Y axis": y_axis,
            "dataset": dataset,
            "Predictions": pred,
            "R square": r_square,
            "Adj R square": adj,
            "Mean square error": mse,
            "time": time,
            "date": date,
            "no_var": no_var,
            "Coefficient and Intercept": coff
        })


# @app.route("/forcast", methods=["POST"])
# def forcast():

#     req_data = request.get_json()
#     dataset = req_data['Dataset']
#     timestamp = req_data['TimeStamp']
#     x_axis = req_data['x_axis']
#     y_axis = req_data['y_axis']
#     start_time = req_data['start_time']
#     end_time = req_data['end_time']
#     df = fc.main(dataset, x_axis, y_axis, timestamp, start_time, end_time)
#     df = df.reset_index()
#     df['index'] = pd.to_datetime(df['index'])
#     df['index'] = df['index'].dt.strftime('%Y-%m-%d')

#     df = df.to_json(orient='records', date_format='iso')
#     print(df)
#     return jsonify({
#         "Dataset": dataset,
#         "x_axis": x_axis,
#         "y_axis": y_axis,
#         "Timestamp": timestamp,
#         "Processed dataset": df,
#     })


@app.route("/predictmanual", methods=["POST"])
def predict_manual():
    req_data = request.get_json()
    model_name = req_data['Model_name']
    model_type = req_data['Model_type']
    x_axis = req_data['x_axis']
    model = req_data['Model']
    predict_data = req_data['predict_data']
    if(model_type == "log_regression"):
        keep = req_data['Keep']
        pred = m.log_pred(model, x_axis, keep, predict_data)
        return jsonify({
            "Modal_name": model_name,
            "Modal_type": model_type,
            "Data": predict_data,
            "prediction": pred
        })
    else:

        model_name = req_data['Model_name']
        x_test = req_data['X_test']
        x_axis = req_data['x_axis']
        y_axis = req_data['y_axis']
        model_type = req_data['Model_type']
        model = req_data['Model']
        predict_data = req_data['predict_data']
        pred, df, r_square, adj, mse = m.model_pred_manual(model, x_axis,
                                                           predict_data)

        pred = pred.to_json(orient='records')
    # df = df.to_json(orient='records')
    # return jsonify({"Model":model,"Model_name":model_name,"X_test":df,"Model_type":model_type,"x_axis":x_axis,"y_axis":y_axis,"R square":r_square,"Adj R Square":adj,"Mean Squared Error":mse})
        return jsonify({
            "Model": model,
            "x_axis": x_axis,
            "Model_name": model_name,
            "Modal_type": model_type,
            "y_axis": y_axis,
            "X_test": x_test,
            "Predictions": pred,
            "R square": r_square,
            "Adj R square": adj,
            "Mean square error": mse,
            "predict_data": predict_data
        })


if __name__ == "__main__":

    app.run(port=5555)
