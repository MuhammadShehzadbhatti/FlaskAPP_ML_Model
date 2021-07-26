import requests
import pickle
import json
import collections
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging
import sys
import random
from datetime import datetime
import math
from collections import defaultdict
import seaborn as sns
import pandas as pd
import numpy as np
from io import StringIO


class MF:

    Regularization = 0.001
    BiasLearnRate = 0.005
    BiasReg = 0.002
    LearnRate = 0
    #LearnRate = 0
    all_layouts_mean = 0
    number_of_ratings = 0
    save_path = '~/Documents'
    item_bias = None
    user_bias = None
    beta = 0.02

    iterations = 0

    def __init__(self, max_iterations, learningRate):
        print('learningRate', learningRate)
        self.user_factors = None
        self.item_factors = None
        self.item_counts = None
        self.item_sum = None
        self.u_inx = None
        self.i_inx = None
        self.user_ids = None
        self.layouts_ids = None
        self.LearnRate = learningRate
        self.all_layouts_mean = 0.0
        self.number_of_ratings = 0
        self.MAX_ITERATIONS = max_iterations
        self.rated_csv = pd.read_csv('calculated_ratings_before_training.csv')
        random.seed(42)

    def initialize_factors(self, ratings, k):

        # get all the user ids from the datafram
        self.user_ids = set(ratings['user_id'].values)
        # get all the layout ids from the datafram
        self.layout_ids = set(ratings['layout_id'].values)

        # creating dictionaries of the user and items and generating the numeric values against to make it simple for numpay
        self.u_inx = {r: i for i, r in enumerate(self.user_ids)}
        self.i_inx = {r: i for i, r in enumerate(self.layout_ids)}
        #print(self.u_inx )
        self.item_factors = np.full((len(self.i_inx), k), (0.01))
        self.user_factors = np.full((len(self.u_inx), k), 0.01)

        self.all_layouts_mean = self.find_mean_ratings(
            ratings)  # also called global bias
        print("self.all_layouts_mean ", self.all_layouts_mean.size)
        # initialization of user and item biases
        self.user_bias = defaultdict(lambda: 0)
        self.item_bias = defaultdict(lambda: 0)

    def find_mean_ratings(self, results):

        avg = results['ratings'].sum() / results.shape[0]
        return avg

    def train(self, ratings_df, k):
        self.initialize_factors(ratings_df, k)

        #print("training matrix factorization at {}".format(datetime.now()))

        train_data, test_data = self.split_data(ratings_df)
        columns = ['user_id', 'layout_id', 'ratings']
        ratings = train_data[columns].values
        test = test_data[columns].values

        # creating list equal to the length of ratings dataframe numpy array
        index_randomized = random.sample(
            range(0, len(ratings)), (len(ratings) - 1))

        for factor in range(k):
            factor_time = datetime.now()
            iterations = 0  # factor initializer

            # print('factor',factor)

            # first time intializing with maximum latest/last and iteration errors

            last_err = sys.maxsize
            max_error = last_err
            iteration_err = sys.maxsize

            finished = False

            while not finished:
                start_time = datetime.now()
                iteration_err = self.stocastic_gradient_descent(
                    factor, index_randomized, ratings)  # SGD optimizer

                iterations += 1
                # print(self.LearnRate)
                #print("epoch in {}, factor={}, iterations={} iteration_err={} LearningRate={}".format(datetime.now() - start_time, factor, iterations, iteration_err, self.LearnRate))
                finished = self.finished(iterations, last_err, iteration_err)
                last_err = iteration_err

            self.save(factor, finished)
            #logging.debug("finished factor {} on f={} i={} err={}".format(factor, datetime.now() - factor_time, iterations, iteration_err))
            print("finished factor {} on time={} iterations={} iteration_err={}".format(
                factor, datetime.now() - factor_time, iterations, iteration_err))
            #print("self.item_factors_dot_self.user_factors {}\n".format(np.dot(self.item_factors, self.user_factors.T) ))
            #print("ratings {}".format(type(ratings) ))

    def split_data(self, ratings):
        #msk = np.random.rand(len(ratings)) < 0.8
        # print(msk)
        #training_df = ratings[msk]
        #validation_df = ratings[~msk]
        training_df, validation_df = train_test_split(
            ratings, test_size=.2, random_state=42)
        return training_df, validation_df

    def finished(self, iterations, last_err, current_err):
        if iterations >= self.MAX_ITERATIONS:
            #print('Finish w iterations: {}, last_err: {}, current_err {}' .format(iterations, last_err, current_err))
            return True
        else:
            self.iterations += 1
            return False

    def stocastic_gradient_descent(self, factor, index_randomized, ratings):

        lr = self.LearnRate
        b_lr = self.BiasLearnRate
        r = self.Regularization
        bias_r = self.BiasReg
       #print('u_inx :',u_inx)
        for inx in index_randomized:
            rating_row = ratings[inx]

            u = self.u_inx[rating_row[0]]  # user's index in dictionary
            i = self.i_inx[rating_row[1]]  # item's index in dictionary
            rating = rating_row[2]

            prediction = self.predict(u, i)

            err = rating - prediction

            self.user_bias[u] += b_lr * (err - bias_r * self.user_bias[u])
            self.item_bias[i] += b_lr * (err - bias_r * self.item_bias[i])

            user_fac = self.user_factors[u][factor]
            item_fac = self.item_factors[i][factor]

            self.user_factors[u][factor] += lr * \
                (err * item_fac - r * user_fac)
            self.item_factors[i][factor] += lr * \
                (err * user_fac - r * item_fac)

        return self.calculate_rmse(ratings, factor)

    def predict(self, user, item):

        avg = self.all_layouts_mean

        item_test = self.item_factors[item]
        user_test = self.user_factors[user]

        pq = np.dot(item_test, user_test)
        b_ui = avg + self.user_bias[user] + self.item_bias[item]

        prediction = b_ui + pq

        if prediction > 5:
            prediction = 5
        elif prediction < 1:
            prediction = 1
        #print('item {} User {} Prediction {}'.format(item, user, prediction))
        return prediction

    def calculate_rmse(self, ratings, factor):
        # print('ratings',ratings)
        def difference(row):
            user = self.u_inx[row[0]]
            item = self.i_inx[row[1]]

#             print('ratings: {} user: {}',ratings[user],user)
#             print('ratings: {}item: {}',ratings[item], item)
            pq = np.dot(
                self.item_factors[item][:factor + 1], self.user_factors[user][:factor + 1].T)
            b_ui = self.all_layouts_mean + \
                self.user_bias[user] + self.item_bias[item]
            prediction = b_ui + pq
            MSE = (prediction - row[2]) ** 2
            return MSE

        squared = np.apply_along_axis(difference, 1, ratings).sum()

        rmse = math.sqrt(squared / ratings.shape[0])

        return rmse

    def save(self, factor, finished):

        save_path = self.save_path + '/'
        if not finished:
            save_path += str(factor) + '/'

        print("saving factors in {}".format(save_path))
        user_bias = {uid: self.user_bias[self.u_inx[uid]]
                     for uid in self.u_inx.keys()}
        item_bias = {iid: self.item_bias[self.i_inx[iid]]
                     for iid in self.i_inx.keys()}

        userFactor = pd.DataFrame(self.user_factors, index=self.user_ids)
        itemFactor = pd.DataFrame(self.item_factors, index=self.layout_ids)

        with open('user_factors.json', 'w') as outfile:
            outfile.write(userFactor.to_json())
        with open('item_factors.json', 'w') as outfile:
            outfile.write(itemFactor.to_json())
        with open('user_bias.data', 'wb') as ub_file:
            pickle.dump(user_bias, ub_file)
        with open('item_bias.data', 'wb') as ub_file:
            pickle.dump(item_bias, ub_file)
        with open('avg.npy', 'wb') as f:
            np.save(f, self.all_layouts_mean)

    #lr_rate = [0.001,0.002,0.01,0.02,0.03,0.04,0.05,0.003,0.004,0.005]

    # for lr in lr_rate:
    # as increasing the iterations error is decreasing but keeping it at 100
    max_iterations = 100
    # by increasing or decreasing the learning rate the megnitude of error(i.e. 0.1060) also increases so keeping it at 0.03 for testing
# lr = 0.03
# mf_obj = MF(max_iterations, lr)
    # print(type(result))
    #fac = [2,4,6,8,10,12,14,16,18,20]
    # for i in fac:
    # magnitude of the error(0.1060) is same from 8 onwards by keeping it as and testing it while keeping teh same params
# mf_obj.train(result, 8)
    # on testing with same params error is 0.1037 but somewhere it's 0.1003 with only one factor maight be it's possible beacuse of a couple of records for testing

    # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:08.049929Z","iopub.execute_input":"2021-05-26T21:27:08.050267Z","iopub.status.idle":"2021-05-26T21:27:08.084118Z","shell.execute_reply.started":"2021-05-26T21:27:08.050233Z","shell.execute_reply":"2021-05-26T21:27:08.082468Z"}}

    # def recommend_items(self, user_id, num=6):

    #     active_user_items = Rating.objects.filter(user_id=user_id).order_by('-rating')[:100]

    #     return self.recommend_items_by_ratings(user_id, active_user_items.values())
    # load_model
# table = pd.pivot_table(result, values='ratings',
#                         index='user_id', columns=['layout_id'])
# df_dum = pd.DataFrame(table.to_records()).T
# type(df_dum)
# df_dum.index.name = 'layouts'
# df_dum

    # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:08.085920Z","iopub.execute_input":"2021-05-26T21:27:08.086397Z","iopub.status.idle":"2021-05-26T21:27:08.092022Z","shell.execute_reply.started":"2021-05-26T21:27:08.086346Z","shell.execute_reply":"2021-05-26T21:27:08.090646Z"}}
# test_result = result

    # %% [code] {"jupyter":{"outputs_hidden":false}}

    # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:50:15.629559Z","iopub.execute_input":"2021-05-26T21:50:15.629953Z","iopub.status.idle":"2021-05-26T21:50:15.666485Z","shell.execute_reply.started":"2021-05-26T21:50:15.629918Z","shell.execute_reply":"2021-05-26T21:50:15.665598Z"}}
