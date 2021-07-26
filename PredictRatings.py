import pickle
import json
import collections
import numpy as np
import pandas as pd


class PredictedRatings_test:

    def __init__(self):
        self.uid = ""
        self.user_bias = None
        self.item_bias = None
        self.avg = None
        self.user_factors = None
        self.item_factors = None
        self.ordered_item_bias = None
        self.model_loaded = None
        self.model_loaded = False
        self.load_model_prediction()
        # self.user_id = []
        self.datf = None
        self.pr_df = pd.DataFrame(
            columns=['layouts', 'predicted_ratings', 'user_id'])

    def load_model_prediction(self):
        with open('user_bias.data', 'rb') as ub_file:
            self.user_bias = pickle.load(ub_file)
        with open('item_bias.data', 'rb') as ub_file:
            self.item_bias = pickle.load(ub_file)
        with open('avg.npy', 'rb') as f:
            self.avg = np.load(f)

        with open('user_factors.json', 'r') as infile:
            self.user_factors = pd.DataFrame(json.load(infile)).T
        with open('item_factors.json', 'r') as infile:
            self.item_factors = pd.DataFrame(json.load(infile)).T
        self.ordered_item_bias = list(collections.OrderedDict(
            sorted(self.item_bias.items())).values())
        self.model_loaded = True
        # print(self.user_factors)

    def recommend_items_by_ratings_prediction(self, uid):
        self.uid = uid
        # self.results = result_analytics

        print("uid: ", self.uid)
        #print("results: ", self.results)
        global pr_df
        recs = {}
        print("self.user_factors.columns", type(self.user_factors.columns))
        # get each user one by one and predict layouts of him
        # if not self.uid in self.user_factors:
        for userId in self.user_factors:

            user = self.user_factors[str(userId)]

            items = self.item_factors

            scores = items.T.dot(user)

            user_bias = 0

            if userId in self.user_bias.keys():
                user_bias = self.user_bias[userId]
            elif int(userId) in self.user_bias.keys():
                user_bias = self.user_bias[int(userId)]
                print(f'it was an int {user_bias}')

            rating = float(user_bias + self.avg)
            scores += rating
            recs = {r[0]: {'prediction': r[1] + float(self.item_bias[r[0]])}
                    for r in zip(scores.index, scores)}
            sorted_items = sorted(
                recs.items(), key=lambda item: -float(item[1]['prediction']))
            k = ""
            val = ""
            k_list = []
            val_list = []
            user_list = []
            for index, tpl in enumerate(sorted_items):
                self.pr_df = self.pr_df.append(
                    {'layouts': tpl[0], 'user_id': userId, 'predicted_ratings': tpl[1]['prediction']}, ignore_index=True)
        # self.pr_df = self.pr_df.groupby(
        #     'layouts').mean('predicted_ratings')

        print("pr_df", self.pr_df.sort_values(
            by=['predicted_ratings'], ascending=False))

        self.pr_df = self.pr_df.sort_values(
            by=['predicted_ratings'], ascending=False)

        # print("pr_df", self.pr_df.sort_values(
        #     by=['predicted_ratings'], ascending=False))
        # self.pr_df['layouts'] = self.pr_df['layouts'].astype(str)

        print('self.pr_df.info():', self.pr_df.info())
        return self.pr_df

    def recommend_items_for_existing_user(self, user_id, active_user_items, num, result_analytics):
       # print(user_id)
        # print(active_user_items.layout_id)
        rated_layouts = set(active_user_items.layout_id)
       # print(rated_layouts)
        recs = {}
       # print("____self.user_factors.columns____",self.user_factors.columns)
        if user_id in self.user_factors.columns:
            # print("str('user_id')", str('user_id'))
            user = self.user_factors[str(user_id)]

            items = self.item_factors

           # print("____self.items____", self.items)
            # print("____user____",user)

            scores = items.T.dot(user)

            # sorting predictions before adding bias
            sorted_scores = scores.sort_values(ascending=False)
            # print("sorted_scores",sorted_scores)
            result = sorted_scores[:num + len(rated_layouts)]
            # result = scores[:num + len(rated_layouts)]
            # print("sorted_scores[:num]",sorted_scores[:num])
            user_bias = 0

            if user_id in self.user_bias.keys():
                user_bias = self.user_bias[user_id]
            elif int(user_id) in self.user_bias.keys():
                user_bias = self.user_bias[int(user_id)]
                print(f'it was an int {user_bias}')

            rating = float(user_bias + self.avg)
            result += rating

            # sorting predictions before adding bias
            # result = result.sort_values(ascending=False)
            check_list = []
            for a in items.columns.tolist():
                print("____self.items", len(items.columns.tolist()))
                if a in rated_layouts:
                    check_list.append(a)
                    if len(check_list) == len(items.columns.tolist()) or len(check_list) == result_analytics.layout_id.nunique():

                        recs = {r[0]: {'prediction': r[1] + float(self.item_bias[r[0]])}
                                for r in zip(result.index, result)}
                        print("____self.items", type(
                            items.columns.tolist()), a)
                        recs = dict(sorted(
                            recs.items(), key=lambda item: -float(item[1]['prediction'])))

                    elif len(check_list) > 0 and len(check_list) < len(items.columns.tolist()) or len(check_list) < result_analytics.layout_id.nunique():

                        recs = {r[0]: {'prediction': r[1] + float(self.item_bias[r[0]])}
                                for r in zip(result.index, result) if r[0] not in rated_layouts}
                        print('recs_before_append', recs)
                        recs = dict(sorted(
                            recs.items(), key=lambda item: -float(item[1]['prediction'])))

                    abc = {r[0]: {'prediction': r[1] + float(self.item_bias[r[0]])}
                           for r in zip(result.index, result) if r[0] in rated_layouts}
                    print('abc_before_append', abc)
                    recs.update(abc)
                    print('recs_after_append', recs)

        # sorted_items = sorted(
        #     recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]
        for key, value in recs.items():
            #print('tpl[0]', type(tpl), index)
            # for k, v in value.values():
            print('tpl[0]', key, value['prediction'])

            self.pr_df = self.pr_df.append(
                {'layouts': key, 'user_id': user_id, 'predicted_ratings': value['prediction']}, ignore_index=True)

        return self.pr_df
