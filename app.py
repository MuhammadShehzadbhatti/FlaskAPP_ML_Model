#from Import_data import ImportData, import_data
import json

from flask.wrappers import Response
from PredictRatings import PredictedRatings_test
from oauth2client.client import GoogleCredentials
#from pydrive.drive import GoogleDrive
#from pydrive.auth import GoogleAuth
from flask import jsonify
from ImportingData import DataImporting
import os
#from google.cloud import bigquery
from ImportData import ImportData
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from analysis import Analysis
from MatrixFactorization import MF


app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'etbd-1-2c366b9f240d.json'

#client = bigquery.Client()


@app.route("/analytics")
def analytics():
    analytics = Analysis.dataAnalysis()
    analytics.to_csv('calculated_ratings_before_training.csv')
    print("calculated_ratings_before_training", analytics)
    return Response(analytics.to_json(orient="records"), mimetype='application/json')


@app.route("/trainsmodel")
def trainsmodel():

    max_iterations = 100
    lr = 0.03
    mf_obj = MF(max_iterations, lr)
    result_analytics = pd.read_csv('calculated_ratings_before_training.csv')
    mf_obj.train(result_analytics, 8)
    # analytics = Analysis.dataAnalysis()
    # analytics.to_csv('calculated_ratings_before_training.csv')
    # print("type(analytics)", type(analytics))
    return Response(result_analytics.to_json(orient="records"), mimetype='application/json')


@app.route("/")
def home():
    # predictedRatings = PredictedRatings_test()
    # predicted_df = predictedRatings.recommend_items_by_ratings_prediction()
    # top_layouts_list = []
    # for i in predicted_df.layouts:
    #     if len(top_layouts_list) < 5 and i not in top_layouts_list:
    #         top_layouts_list.append(i)
    # print('pr_df', predicted_df)
    # print('top_layouts_list', top_layouts_list)
    return ("TESTING APP")


@app.route("/columns")
def columns():
    #importDataObj = ImportData()
    #print("home", type(home()))
    data_importing = DataImporting()
    data = data_importing.csv_data_importing()

    # print("data", type(data))
    # print("data columns", data)

    # ImportDataObj = ImportData()
    # ll = ImportDataObj.import_bq_data(data)

    # print("llabc", type(ll))
    newlist = []
    for col in data.columns:
        newlist.append(col)
        print('newlist', newlist)
        # return col
    # quey_res = qry()
    # for row in quey_res:
    #     return (print(f'{row.user_pseudo_id}: {row.event_name}'))
    return jsonify(newlist)


@app.route("/rows/<uid>", methods=['GET'])
def rows(uid):
    predictedRatings = PredictedRatings_test()
    # predictedRatings = PredictedRatings_test()
    result_analytics = pd.read_csv('calculated_ratings_before_training.csv')
    print("result_analytics", result_analytics.user_id.tolist())
    predicted_df = pd.DataFrame(columns=result_analytics.columns)
    print('predicted_df.count()',
          result_analytics.loc[result_analytics.user_id == uid].count())
    if (result_analytics.loc[(result_analytics.user_id == uid), 'layout_id'].count()) > 0:
        print('uid....', uid)

        layouts_used_by_user = result_analytics.loc[result_analytics.user_id == uid]
        print('layouts_used_by_user', layouts_used_by_user)
        predicted_df = predictedRatings.recommend_items_for_existing_user(
            uid, layouts_used_by_user, result_analytics.layout_id.nunique(), result_analytics)

        print('predicted ratings', result_analytics)

    else:
        predicted_df = predictedRatings.recommend_items_by_ratings_prediction(
            uid)
        print('type(predicted_df)', predicted_df.index)

    #top_layouts_list = []
    # for i in predicted_df.layouts:
    #     if len(top_layouts_list) < 5 and i not in top_layouts_list:
    #         top_layouts_list.append(i)
    print('pr_df', predicted_df)
    #print('top_layouts_list', top_layouts_list)
    #df = predicted_df[~(predicted_df.index.duplicated(keep='first'))]
    #predicted_df.drop_duplicates(subset=['layouts'], keep='first')
    #print('df', df)
    print('predicted_df', predicted_df)

    return Response(predicted_df.to_json(orient="records"), mimetype='application/json')


@app.route("/post", methods=['GET', 'POST'])
def ppost():
    global value1
    global result
    global df
    if request.method == "POST":
        val1 = request.form['uid']
        print("request_Form2: ", val1)
        # predictedRatings = PredictedRatings_test()
        # predicted_df = predictedRatings.recommend_items_by_ratings_prediction()
        # top_layouts_list = []
        # for i in predicted_df.layouts:
        #     if len(top_layouts_list) < 5 and i not in top_layouts_list:
        #         top_layouts_list.append(i)
        # print('pr_df', predicted_df)
        # print('top_layouts_list', top_layouts_list)
        # df = predicted_df.drop_duplicates(subset=['layouts'], keep='first')
        # print('df', df)
        # print('predicted_df', predicted_df)

        # return Response(df.to_json(orient="records"), mimetype='application/json')
        return request.form['uid']

    else:
        predictedRatings = PredictedRatings_test()
        predicted_df = predictedRatings.recommend_items_by_ratings_prediction()
        top_layouts_list = []
        for i in predicted_df.layouts:
            if len(top_layouts_list) < 5 and i not in top_layouts_list:
                top_layouts_list.append(i)
        print('pr_df', predicted_df)
        print('top_layouts_list/n', top_layouts_list)
        df = predicted_df.drop_duplicates(subset=['layouts'], keep='first')
        print('df', df)
        print('predicted_df', predicted_df)
        # result = df.to_json(orient="records")

        # parsed = json.loads(result)
        # result = json.dumps(parsed, indent=4)
        return Response(df.to_json(orient="records"), mimetype='application/json')
    return Response(df.to_json(orient="records"), mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True)
