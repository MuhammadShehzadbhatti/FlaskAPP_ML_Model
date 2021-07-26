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
from io import StringIO
import matplotlib.pyplot as plt


class Analysis:

    def dataAnalysis():

        df1 = pd.read_csv('bq-results-20210204-100630-jm02wiso21h7.csv')
        df2 = pd.read_csv('bq-results-20210412-115757-d38jg8t5bj3v.csv')

        df = df2.append([df1], ignore_index=True)
        print('df.columns', df.columns)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.670093Z","iopub.execute_input":"2021-05-26T21:27:00.670705Z","iopub.status.idle":"2021-05-26T21:27:00.716569Z","shell.execute_reply.started":"2021-05-26T21:27:00.670670Z","shell.execute_reply":"2021-05-26T21:27:00.715341Z"}}
        dup_timestamp = df[df.duplicated(subset=['event_timestamp'])]
        print("dup_timestamp: ", dup_timestamp)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.718097Z","iopub.execute_input":"2021-05-26T21:27:00.718422Z","iopub.status.idle":"2021-05-26T21:27:00.751758Z","shell.execute_reply.started":"2021-05-26T21:27:00.718391Z","shell.execute_reply":"2021-05-26T21:27:00.750363Z"}}
        print("df.info()", df.info())

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.753456Z","iopub.execute_input":"2021-05-26T21:27:00.753809Z","iopub.status.idle":"2021-05-26T21:27:00.763175Z","shell.execute_reply.started":"2021-05-26T21:27:00.753776Z","shell.execute_reply":"2021-05-26T21:27:00.762208Z"}}
        df.drop(['Column1', 'Column2', 'Column3', 'Column4', 'Home_Button_Visibility', 'Category_Button_Visibility', 'Map_Button_Visibility',
                 'ITEM_RELEASE', 'ITEM_RATING', 'ITEM_VOTE_COUNT', 'ITEM_ORIGINAL_LANGUAGE', 'ITEM_GENRE', 'COLUMN', 'items'], axis='columns', inplace=True)

        print('df.columns_2', df.columns)
        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.764777Z","iopub.execute_input":"2021-05-26T21:27:00.765084Z","iopub.status.idle":"2021-05-26T21:27:00.779803Z","shell.execute_reply.started":"2021-05-26T21:27:00.765055Z","shell.execute_reply":"2021-05-26T21:27:00.778381Z"}}
        print("type(df): ", type(df))
        print("df.event_date.unique()", df.loc[:, 'event_date'].unique())
        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.781877Z","iopub.execute_input":"2021-05-26T21:27:00.782576Z","iopub.status.idle":"2021-05-26T21:27:00.808983Z","shell.execute_reply.started":"2021-05-26T21:27:00.782525Z","shell.execute_reply":"2021-05-26T21:27:00.807564Z"}}
        df = df.loc[(df.event_date >= '2021-02-03'), :]
        df.info()

        # %% [code] {"execution":{"iopub.status.busy":"2021-05-26T21:27:00.810736Z","iopub.execute_input":"2021-05-26T21:27:00.811176Z","iopub.status.idle":"2021-05-26T21:27:00.851027Z","shell.execute_reply.started":"2021-05-26T21:27:00.811133Z","shell.execute_reply":"2021-05-26T21:27:00.849694Z"}}
        df

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.852774Z","iopub.execute_input":"2021-05-26T21:27:00.853223Z","iopub.status.idle":"2021-05-26T21:27:00.886797Z","shell.execute_reply.started":"2021-05-26T21:27:00.853177Z","shell.execute_reply":"2021-05-26T21:27:00.885485Z"}}
        df.describe()

        # %% [markdown]
        # ## cleansing

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.888871Z","iopub.execute_input":"2021-05-26T21:27:00.889369Z","iopub.status.idle":"2021-05-26T21:27:00.915711Z","shell.execute_reply.started":"2021-05-26T21:27:00.889292Z","shell.execute_reply":"2021-05-26T21:27:00.914501Z"}}
        nonNull_direbase_screen = df[(df.firebase_screen.notnull()) & (
            df.event_name.notnull()) & (df.event_name != 'my_tag_event')]
        nonNull_direbase_screen.head()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:00.917467Z","iopub.execute_input":"2021-05-26T21:27:00.917849Z","iopub.status.idle":"2021-05-26T21:27:01.383499Z","shell.execute_reply.started":"2021-05-26T21:27:00.917811Z","shell.execute_reply":"2021-05-26T21:27:01.381978Z"}}
        nonNull_direbase_screen.groupby('event_timestamp').filter(
            lambda x: len(x['event_name'].unique()) > 1)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.385020Z","iopub.execute_input":"2021-05-26T21:27:01.385329Z","iopub.status.idle":"2021-05-26T21:27:01.407584Z","shell.execute_reply.started":"2021-05-26T21:27:01.385280Z","shell.execute_reply":"2021-05-26T21:27:01.406436Z"}}
        df.head()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.409198Z","iopub.execute_input":"2021-05-26T21:27:01.409586Z","iopub.status.idle":"2021-05-26T21:27:01.438507Z","shell.execute_reply.started":"2021-05-26T21:27:01.409516Z","shell.execute_reply":"2021-05-26T21:27:01.437566Z"}}
        df['engagement_time_seconds'] = df.engagement_time_msec/1000
        df[(~df.engagement_time_seconds.isna())].head()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.439818Z","iopub.execute_input":"2021-05-26T21:27:01.440140Z","iopub.status.idle":"2021-05-26T21:27:01.446659Z","shell.execute_reply.started":"2021-05-26T21:27:01.440109Z","shell.execute_reply":"2021-05-26T21:27:01.445407Z"}}
        df.drop(['engagement_time_msec'], axis='columns', inplace=True)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.448079Z","iopub.execute_input":"2021-05-26T21:27:01.448445Z","iopub.status.idle":"2021-05-26T21:27:01.466543Z","shell.execute_reply.started":"2021-05-26T21:27:01.448412Z","shell.execute_reply":"2021-05-26T21:27:01.465542Z"}}
        df.loc[(df['event_name'] == 'screen_view'),
               'engagement_time_seconds'].sum()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.467968Z","iopub.execute_input":"2021-05-26T21:27:01.468293Z","iopub.status.idle":"2021-05-26T21:27:01.482456Z","shell.execute_reply.started":"2021-05-26T21:27:01.468251Z","shell.execute_reply":"2021-05-26T21:27:01.481560Z"}}
        df.loc[(df['event_name'] == 'user_engagement'),
               'engagement_time_seconds'].sum()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.484197Z","iopub.execute_input":"2021-05-26T21:27:01.484842Z","iopub.status.idle":"2021-05-26T21:27:01.498092Z","shell.execute_reply.started":"2021-05-26T21:27:01.484807Z","shell.execute_reply":"2021-05-26T21:27:01.497011Z"}}
        df['firebase_screen'].unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.499441Z","iopub.execute_input":"2021-05-26T21:27:01.499926Z","iopub.status.idle":"2021-05-26T21:27:01.527051Z","shell.execute_reply.started":"2021-05-26T21:27:01.499877Z","shell.execute_reply":"2021-05-26T21:27:01.526089Z"}}
        df.info()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.528855Z","iopub.execute_input":"2021-05-26T21:27:01.529191Z","iopub.status.idle":"2021-05-26T21:27:01.552141Z","shell.execute_reply.started":"2021-05-26T21:27:01.529159Z","shell.execute_reply":"2021-05-26T21:27:01.551269Z"}}
        df.head()

        # %% [markdown]
        # * ## cleaning firebase_screen_class

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.553419Z","iopub.execute_input":"2021-05-26T21:27:01.553930Z","iopub.status.idle":"2021-05-26T21:27:01.568401Z","shell.execute_reply.started":"2021-05-26T21:27:01.553878Z","shell.execute_reply":"2021-05-26T21:27:01.567466Z"}}
        df.firebase_screen_class.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.569680Z","iopub.execute_input":"2021-05-26T21:27:01.570233Z","iopub.status.idle":"2021-05-26T21:27:01.596703Z","shell.execute_reply.started":"2021-05-26T21:27:01.570186Z","shell.execute_reply":"2021-05-26T21:27:01.595819Z"}}
        df = df.loc[(df.firebase_screen_class == 'MainActivity_V1'), :]
        df.info()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.598014Z","iopub.execute_input":"2021-05-26T21:27:01.598565Z","iopub.status.idle":"2021-05-26T21:27:01.610989Z","shell.execute_reply.started":"2021-05-26T21:27:01.598520Z","shell.execute_reply":"2021-05-26T21:27:01.609710Z"}}
        df[(df.firebase_screen_class != 'MainActivity_V1')]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.612882Z","iopub.execute_input":"2021-05-26T21:27:01.613378Z","iopub.status.idle":"2021-05-26T21:27:01.629128Z","shell.execute_reply.started":"2021-05-26T21:27:01.613330Z","shell.execute_reply":"2021-05-26T21:27:01.627966Z"}}
        df.firebase_screen_class.value_counts()

        # %% [markdown]
        # * ## cleaning firebase_screen

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.630806Z","iopub.execute_input":"2021-05-26T21:27:01.631574Z","iopub.status.idle":"2021-05-26T21:27:01.649038Z","shell.execute_reply.started":"2021-05-26T21:27:01.631509Z","shell.execute_reply":"2021-05-26T21:27:01.647839Z"}}
        df.firebase_screen.value_counts()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.650771Z","iopub.execute_input":"2021-05-26T21:27:01.651152Z","iopub.status.idle":"2021-05-26T21:27:01.672445Z","shell.execute_reply.started":"2021-05-26T21:27:01.651117Z","shell.execute_reply":"2021-05-26T21:27:01.671568Z"}}
        # drop unnecessary values in firebase screen
        firebase_screen_nan_index = df[(df['firebase_screen'].isna()) | (df['firebase_screen'] == 'nan') | (df['firebase_screen'] == 'productDetailsFragment') | (df['firebase_screen'] == 'HomeFragment') |
                                       (df['firebase_screen'] == 'Maps') | (df['firebase_screen'] == 'ProductsCategory') | (df['firebase_screen'] == 'nullTab')].index
        df.drop(index=firebase_screen_nan_index, inplace=True)
        df.firebase_screen.unique()

        # %% [markdown]
        # * ## Screen_View Event

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.673757Z","iopub.execute_input":"2021-05-26T21:27:01.674062Z","iopub.status.idle":"2021-05-26T21:27:01.693334Z","shell.execute_reply.started":"2021-05-26T21:27:01.674031Z","shell.execute_reply":"2021-05-26T21:27:01.692476Z"}}
        df.info()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.694778Z","iopub.execute_input":"2021-05-26T21:27:01.695078Z","iopub.status.idle":"2021-05-26T21:27:01.714077Z","shell.execute_reply.started":"2021-05-26T21:27:01.695049Z","shell.execute_reply":"2021-05-26T21:27:01.712914Z"}}
        screen_view_df = df.loc[(df['event_name'] == 'screen_view') | (df['event_name'] == 'user_engagement'), [
            'user_id', 'event_name', 'firebase_screen', 'engagement_time_seconds']]
        screen_view_df.head()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.715650Z","iopub.execute_input":"2021-05-26T21:27:01.715947Z","iopub.status.idle":"2021-05-26T21:27:01.729408Z","shell.execute_reply.started":"2021-05-26T21:27:01.715917Z","shell.execute_reply":"2021-05-26T21:27:01.728031Z"}}
        screen_view_df[(screen_view_df.event_name == 'user_engagement')
                       & (screen_view_df.engagement_time_seconds.isna())]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.731033Z","iopub.execute_input":"2021-05-26T21:27:01.731540Z","iopub.status.idle":"2021-05-26T21:27:01.755936Z","shell.execute_reply.started":"2021-05-26T21:27:01.731490Z","shell.execute_reply":"2021-05-26T21:27:01.754490Z"}}
        screen_view_df[(screen_view_df.event_name == 'screen_view') &
                       (screen_view_df.engagement_time_seconds.isna())]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.758057Z","iopub.execute_input":"2021-05-26T21:27:01.758552Z","iopub.status.idle":"2021-05-26T21:27:01.764966Z","shell.execute_reply.started":"2021-05-26T21:27:01.758501Z","shell.execute_reply":"2021-05-26T21:27:01.763342Z"}}
        #screen_view_df.drop(index = screen_view_df[(screen_view_df.engagement_time_seconds.isna())].index,inplace=True)
        # screen_view_df

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.767006Z","iopub.execute_input":"2021-05-26T21:27:01.767510Z","iopub.status.idle":"2021-05-26T21:27:01.781065Z","shell.execute_reply.started":"2021-05-26T21:27:01.767459Z","shell.execute_reply":"2021-05-26T21:27:01.779546Z"}}
        screen_view_df.loc[(screen_view_df['event_name'] ==
                            'screen_view'), 'engagement_time_seconds'].sum()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.783096Z","iopub.execute_input":"2021-05-26T21:27:01.783751Z","iopub.status.idle":"2021-05-26T21:27:01.797266Z","shell.execute_reply.started":"2021-05-26T21:27:01.783687Z","shell.execute_reply":"2021-05-26T21:27:01.795448Z"}}
        screen_view_df.loc[(screen_view_df['event_name'] ==
                            'user_engagement'), 'engagement_time_seconds'].sum()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:01.799103Z","iopub.execute_input":"2021-05-26T21:27:01.799487Z","iopub.status.idle":"2021-05-26T21:27:02.417547Z","shell.execute_reply.started":"2021-05-26T21:27:01.799452Z","shell.execute_reply":"2021-05-26T21:27:02.416296Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        screen_view_df[(screen_view_df['event_name'] == 'screen_view')].groupby(['user_id', 'firebase_screen']).sum()[
            'engagement_time_seconds'].unstack().plot(kind='bar', ax=ax, alpha=0.75, rot=50, title='Time spent on screen_view event')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.419183Z","iopub.execute_input":"2021-05-26T21:27:02.419551Z","iopub.status.idle":"2021-05-26T21:27:02.810853Z","shell.execute_reply.started":"2021-05-26T21:27:02.419514Z","shell.execute_reply":"2021-05-26T21:27:02.809205Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        screen_view_df[(screen_view_df['event_name'] == 'user_engagement')].groupby(['user_id', 'firebase_screen']).sum()[
            'engagement_time_seconds'].unstack().plot(kind='bar', ax=ax, alpha=0.75, rot=50, title='Time spent on user_engagement event')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.812538Z","iopub.execute_input":"2021-05-26T21:27:02.812875Z","iopub.status.idle":"2021-05-26T21:27:02.832120Z","shell.execute_reply.started":"2021-05-26T21:27:02.812841Z","shell.execute_reply":"2021-05-26T21:27:02.830819Z"}}
        screen_view_df[(screen_view_df['event_name'] == 'screen_view')].groupby(
            ['user_id', 'firebase_screen']).sum()['engagement_time_seconds']

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.834329Z","iopub.execute_input":"2021-05-26T21:27:02.834699Z","iopub.status.idle":"2021-05-26T21:27:02.854428Z","shell.execute_reply.started":"2021-05-26T21:27:02.834655Z","shell.execute_reply":"2021-05-26T21:27:02.852778Z"}}
        screen_view_df[(screen_view_df['event_name'] == 'user_engagement')].groupby(
            ['user_id', 'firebase_screen']).sum()['engagement_time_seconds']

        # %% [markdown]
        # #### user_engagement is shows the engagment of the users with particular layout/screen and the time
        # #### screen_view is the event that causes the occurance of user_engagement event

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.856441Z","iopub.execute_input":"2021-05-26T21:27:02.856834Z","iopub.status.idle":"2021-05-26T21:27:02.882222Z","shell.execute_reply.started":"2021-05-26T21:27:02.856799Z","shell.execute_reply":"2021-05-26T21:27:02.880744Z"}}
        screen_view_df['total_screen_time'] = screen_view_df[(screen_view_df['event_name'] == 'user_engagement')].groupby(
            ['user_id', 'firebase_screen'])['engagement_time_seconds'].transform('sum')
        screen_view_df.head()

        # %% [markdown]
        # ### removing duplicate rows and droping "time_seconds" column

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.888224Z","iopub.execute_input":"2021-05-26T21:27:02.888675Z","iopub.status.idle":"2021-05-26T21:27:02.913831Z","shell.execute_reply.started":"2021-05-26T21:27:02.888630Z","shell.execute_reply":"2021-05-26T21:27:02.912111Z"}}
        screen_view = screen_view_df[(screen_view_df['event_name'] == 'user_engagement')].drop_duplicates(
            subset=['user_id', 'firebase_screen'], keep='first')
        screen_view.drop(['engagement_time_seconds'],
                         axis='columns', inplace=True)
        screen_view.head()

        # %% [markdown]
        # ## including weight = 50 and multiplying with the time spent on each screen

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.915999Z","iopub.execute_input":"2021-05-26T21:27:02.916449Z","iopub.status.idle":"2021-05-26T21:27:02.943031Z","shell.execute_reply.started":"2021-05-26T21:27:02.916408Z","shell.execute_reply":"2021-05-26T21:27:02.941503Z"}}
        screen_view['unscalled_weights'] = screen_view.total_screen_time * 50
        screen_view.drop(['total_screen_time'], axis='columns', inplace=True)
        screen_view.head()

        # %% [markdown]
        # #### screen_view is our final data for event screen_event

        # %% [markdown]
        # * ## Select_content Event

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.945055Z","iopub.execute_input":"2021-05-26T21:27:02.945523Z","iopub.status.idle":"2021-05-26T21:27:02.958788Z","shell.execute_reply.started":"2021-05-26T21:27:02.945486Z","shell.execute_reply":"2021-05-26T21:27:02.957798Z"}}
        select_content_df = df.loc[(df['content'] != 'Three_Col_Button') & (df['content'] != 'Map_Button') & (df['content'] != 'Home_Button') & (
            df['content'] != 'action_signOut') & (df['event_name'] == 'select_content'), ['user_id', 'event_name', 'firebase_screen', 'content', 'content_type']]
        select_content_df.firebase_screen.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.959895Z","iopub.execute_input":"2021-05-26T21:27:02.960171Z","iopub.status.idle":"2021-05-26T21:27:02.989261Z","shell.execute_reply.started":"2021-05-26T21:27:02.960143Z","shell.execute_reply":"2021-05-26T21:27:02.987884Z"}}
        select_content_df = select_content_df.replace(
            {'content': {'L1_Button': 'L1', 'L2_Button': 'L2', 'L3_Button': 'L3', 'L4_Button': 'L4', 'L5_Button': 'L5'}})
        select_content_df

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:02.990940Z","iopub.execute_input":"2021-05-26T21:27:02.991254Z","iopub.status.idle":"2021-05-26T21:27:03.009976Z","shell.execute_reply.started":"2021-05-26T21:27:02.991224Z","shell.execute_reply":"2021-05-26T21:27:03.008654Z"}}
        select_content_df[select_content_df.content.isna()]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.011556Z","iopub.execute_input":"2021-05-26T21:27:03.012064Z","iopub.status.idle":"2021-05-26T21:27:03.026189Z","shell.execute_reply.started":"2021-05-26T21:27:03.012026Z","shell.execute_reply":"2021-05-26T21:27:03.025071Z"}}
        select_content_df.content.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.028212Z","iopub.execute_input":"2021-05-26T21:27:03.028591Z","iopub.status.idle":"2021-05-26T21:27:03.047579Z","shell.execute_reply.started":"2021-05-26T21:27:03.028556Z","shell.execute_reply":"2021-05-26T21:27:03.046690Z"}}
        select_content_df.user_id.unique()

        # %% [markdown]
        # ### droping unnecessary rows and chnaging values

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.048714Z","iopub.execute_input":"2021-05-26T21:27:03.049037Z","iopub.status.idle":"2021-05-26T21:27:03.063588Z","shell.execute_reply.started":"2021-05-26T21:27:03.049006Z","shell.execute_reply":"2021-05-26T21:27:03.062152Z"}}
        select_content_df.loc[(select_content_df.content == 'gridLayoutHorizontal1') | (select_content_df.content == 'gridLayoutHorizontal2') | (select_content_df.content == 'gridLayoutHorizontal3')
                              | (select_content_df.content == 'gridLayoutHorizontal4') | (select_content_df.content == 'gridLayoutHorizontal5'), 'content_type'] = 'Overflow_menu'

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.065393Z","iopub.execute_input":"2021-05-26T21:27:03.065810Z","iopub.status.idle":"2021-05-26T21:27:03.080934Z","shell.execute_reply.started":"2021-05-26T21:27:03.065775Z","shell.execute_reply":"2021-05-26T21:27:03.079695Z"}}
        select_content_df.content.value_counts()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.082355Z","iopub.execute_input":"2021-05-26T21:27:03.082672Z","iopub.status.idle":"2021-05-26T21:27:03.100418Z","shell.execute_reply.started":"2021-05-26T21:27:03.082641Z","shell.execute_reply":"2021-05-26T21:27:03.098417Z"}}
        select_content_df.content_type.value_counts()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.103274Z","iopub.execute_input":"2021-05-26T21:27:03.103871Z","iopub.status.idle":"2021-05-26T21:27:03.116042Z","shell.execute_reply.started":"2021-05-26T21:27:03.103814Z","shell.execute_reply":"2021-05-26T21:27:03.114525Z"}}
        select_content_df.event_name.value_counts()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.117933Z","iopub.execute_input":"2021-05-26T21:27:03.118426Z","iopub.status.idle":"2021-05-26T21:27:03.135323Z","shell.execute_reply.started":"2021-05-26T21:27:03.118375Z","shell.execute_reply":"2021-05-26T21:27:03.133826Z"}}
        select_content_df.firebase_screen.value_counts()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.137156Z","iopub.execute_input":"2021-05-26T21:27:03.137626Z","iopub.status.idle":"2021-05-26T21:27:03.508131Z","shell.execute_reply.started":"2021-05-26T21:27:03.137579Z","shell.execute_reply":"2021-05-26T21:27:03.504077Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        select_content_df.groupby(['user_id', 'firebase_screen']).count()['content'].unstack().plot(
            kind='bar', ax=ax, alpha=0.75, rot=50, title='Number of target screens selection from current screen, "content==target screen"')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.510207Z","iopub.execute_input":"2021-05-26T21:27:03.510563Z","iopub.status.idle":"2021-05-26T21:27:03.527251Z","shell.execute_reply.started":"2021-05-26T21:27:03.510532Z","shell.execute_reply":"2021-05-26T21:27:03.526412Z"}}
        # number of times layouts are selected or the content is target screen that is selected from firebase_screen i.e., current screen
        select_content_df['total_content_counts'] = select_content_df.groupby(
            ['user_id', 'content'])['content'].transform('count')
        select_content_df.head()

        # %% [markdown]
        # * #### drop duplicates and keep the counts counts of events

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.528719Z","iopub.execute_input":"2021-05-26T21:27:03.529033Z","iopub.status.idle":"2021-05-26T21:27:03.536820Z","shell.execute_reply.started":"2021-05-26T21:27:03.529004Z","shell.execute_reply":"2021-05-26T21:27:03.536063Z"}}
        select_content_df.user_id.value_counts()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.538072Z","iopub.execute_input":"2021-05-26T21:27:03.538569Z","iopub.status.idle":"2021-05-26T21:27:03.563465Z","shell.execute_reply.started":"2021-05-26T21:27:03.538522Z","shell.execute_reply":"2021-05-26T21:27:03.562647Z"}}
        select_content_df[(select_content_df.user_id == 'TEsMT2ssaIOeeYroq47ul7Byzub2') & (
            select_content_df.content == 'L3')]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.564598Z","iopub.execute_input":"2021-05-26T21:27:03.564893Z","iopub.status.idle":"2021-05-26T21:27:03.584265Z","shell.execute_reply.started":"2021-05-26T21:27:03.564866Z","shell.execute_reply":"2021-05-26T21:27:03.582524Z"}}
        content_selection = select_content_df.drop_duplicates(
            subset=['user_id', 'content'], keep='first')
        content_selection.user_id.value_counts()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.586191Z","iopub.execute_input":"2021-05-26T21:27:03.586642Z","iopub.status.idle":"2021-05-26T21:27:03.598017Z","shell.execute_reply.started":"2021-05-26T21:27:03.586605Z","shell.execute_reply":"2021-05-26T21:27:03.596558Z"}}
        content_selection.total_content_counts.sum()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.599487Z","iopub.execute_input":"2021-05-26T21:27:03.600041Z","iopub.status.idle":"2021-05-26T21:27:03.611285Z","shell.execute_reply.started":"2021-05-26T21:27:03.599994Z","shell.execute_reply":"2021-05-26T21:27:03.609727Z"}}
        content_selection.total_content_counts.count()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.612947Z","iopub.execute_input":"2021-05-26T21:27:03.613380Z","iopub.status.idle":"2021-05-26T21:27:03.637252Z","shell.execute_reply.started":"2021-05-26T21:27:03.613344Z","shell.execute_reply":"2021-05-26T21:27:03.635874Z"}}
        content_selection

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:03.638888Z","iopub.execute_input":"2021-05-26T21:27:03.639182Z","iopub.status.idle":"2021-05-26T21:27:04.144199Z","shell.execute_reply.started":"2021-05-26T21:27:03.639154Z","shell.execute_reply":"2021-05-26T21:27:04.143387Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        content_selection.groupby(['firebase_screen', 'content']).sum()['total_content_counts'].unstack().plot(
            kind='bar', ax=ax, alpha=0.75, rot=50, title='Number of times target screen selected from current screen')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.145710Z","iopub.execute_input":"2021-05-26T21:27:04.146275Z","iopub.status.idle":"2021-05-26T21:27:04.159536Z","shell.execute_reply.started":"2021-05-26T21:27:04.146222Z","shell.execute_reply":"2021-05-26T21:27:04.158431Z"}}
        content_selection.groupby(['firebase_screen', 'content']).sum()[
            'total_content_counts']

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.161550Z","iopub.execute_input":"2021-05-26T21:27:04.162018Z","iopub.status.idle":"2021-05-26T21:27:04.458336Z","shell.execute_reply.started":"2021-05-26T21:27:04.161967Z","shell.execute_reply":"2021-05-26T21:27:04.457160Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        content_selection.groupby(['content', 'content_type']).sum()['total_content_counts'].unstack().plot(
            kind='bar', ax=ax, alpha=0.75, rot=50, title='Number of times target screen selected from current screen')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.459762Z","iopub.execute_input":"2021-05-26T21:27:04.460076Z","iopub.status.idle":"2021-05-26T21:27:04.475914Z","shell.execute_reply.started":"2021-05-26T21:27:04.460042Z","shell.execute_reply":"2021-05-26T21:27:04.474215Z"}}
        content_selection.head()

        # %% [markdown]
        # ## including weight = 20 and multiplying number of time select_content event occured

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.477555Z","iopub.execute_input":"2021-05-26T21:27:04.477868Z","iopub.status.idle":"2021-05-26T21:27:04.499323Z","shell.execute_reply.started":"2021-05-26T21:27:04.477839Z","shell.execute_reply":"2021-05-26T21:27:04.498523Z"}}

        content_selection['unscalled_weights'] = content_selection.total_content_counts * 20

        content_selection.drop(['firebase_screen', 'content_type',
                                'total_content_counts'], axis='columns', inplace=True)
        content_selection.head()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.500781Z","iopub.execute_input":"2021-05-26T21:27:04.501089Z","iopub.status.idle":"2021-05-26T21:27:04.514248Z","shell.execute_reply.started":"2021-05-26T21:27:04.501058Z","shell.execute_reply":"2021-05-26T21:27:04.513131Z"}}
        content_selection = content_selection.rename(
            columns={"content": "firebase_screen"})
        content_selection.head()

        # %% [markdown]
        # ### content_selection is our final df for select_content event

        # %% [markdown]
        # * ## Select_Item Event

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.515891Z","iopub.execute_input":"2021-05-26T21:27:04.516300Z","iopub.status.idle":"2021-05-26T21:27:04.538609Z","shell.execute_reply.started":"2021-05-26T21:27:04.516267Z","shell.execute_reply":"2021-05-26T21:27:04.537241Z"}}
        df.head()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.540388Z","iopub.execute_input":"2021-05-26T21:27:04.540739Z","iopub.status.idle":"2021-05-26T21:27:04.554753Z","shell.execute_reply.started":"2021-05-26T21:27:04.540707Z","shell.execute_reply":"2021-05-26T21:27:04.553447Z"}}
        df.firebase_screen.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.556801Z","iopub.execute_input":"2021-05-26T21:27:04.557193Z","iopub.status.idle":"2021-05-26T21:27:04.569051Z","shell.execute_reply.started":"2021-05-26T21:27:04.557143Z","shell.execute_reply":"2021-05-26T21:27:04.567971Z"}}
        df.firebase_screen.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.570708Z","iopub.execute_input":"2021-05-26T21:27:04.570996Z","iopub.status.idle":"2021-05-26T21:27:04.583510Z","shell.execute_reply.started":"2021-05-26T21:27:04.570969Z","shell.execute_reply":"2021-05-26T21:27:04.582350Z"}}
        df.item_name.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.584792Z","iopub.execute_input":"2021-05-26T21:27:04.585250Z","iopub.status.idle":"2021-05-26T21:27:04.597815Z","shell.execute_reply.started":"2021-05-26T21:27:04.585207Z","shell.execute_reply":"2021-05-26T21:27:04.596555Z"}}
        df.columns

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.600021Z","iopub.execute_input":"2021-05-26T21:27:04.600378Z","iopub.status.idle":"2021-05-26T21:27:04.619709Z","shell.execute_reply.started":"2021-05-26T21:27:04.600340Z","shell.execute_reply":"2021-05-26T21:27:04.618446Z"}}
        select_item_df = df.loc[(df.event_name == 'select_item') & (df.item_name.notnull()), [
            'user_id', 'event_date', 'event_name', 'firebase_screen', 'item_name', 'ITEM_POSITION_LIST']]
        select_item_df.count()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.621174Z","iopub.execute_input":"2021-05-26T21:27:04.621531Z","iopub.status.idle":"2021-05-26T21:27:04.637155Z","shell.execute_reply.started":"2021-05-26T21:27:04.621499Z","shell.execute_reply":"2021-05-26T21:27:04.635946Z"}}
        select_item_df.item_name.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.638831Z","iopub.execute_input":"2021-05-26T21:27:04.639518Z","iopub.status.idle":"2021-05-26T21:27:04.651248Z","shell.execute_reply.started":"2021-05-26T21:27:04.639459Z","shell.execute_reply":"2021-05-26T21:27:04.650123Z"}}
        select_item_df.user_id.unique()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.652823Z","iopub.execute_input":"2021-05-26T21:27:04.653467Z","iopub.status.idle":"2021-05-26T21:27:04.980092Z","shell.execute_reply.started":"2021-05-26T21:27:04.653420Z","shell.execute_reply":"2021-05-26T21:27:04.979045Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        select_item_df.groupby(['user_id', 'firebase_screen']).count()['item_name'].unstack().plot(
            kind='bar', ax=ax, alpha=0.75, rot=50, title='Number of times, items selected by each user on each layout/screen')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.981653Z","iopub.execute_input":"2021-05-26T21:27:04.982267Z","iopub.status.idle":"2021-05-26T21:27:04.991793Z","shell.execute_reply.started":"2021-05-26T21:27:04.982219Z","shell.execute_reply":"2021-05-26T21:27:04.990540Z"}}
        select_item_df.dtypes

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:04.993645Z","iopub.execute_input":"2021-05-26T21:27:04.993990Z","iopub.status.idle":"2021-05-26T21:27:05.617996Z","shell.execute_reply.started":"2021-05-26T21:27:04.993957Z","shell.execute_reply":"2021-05-26T21:27:05.616562Z"}}
        select_item_df.plot('item_name', 'ITEM_POSITION_LIST', kind='bar', alpha=0.75,
                            rot=90, figsize=(20, 7), title='Items selection at certain positions')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:05.619639Z","iopub.execute_input":"2021-05-26T21:27:05.619947Z","iopub.status.idle":"2021-05-26T21:27:05.640778Z","shell.execute_reply.started":"2021-05-26T21:27:05.619917Z","shell.execute_reply":"2021-05-26T21:27:05.639558Z"}}
        select_item_df['total_item_counts'] = select_item_df.groupby(
            ['user_id', 'firebase_screen', 'item_name'])['item_name'].transform('count')
        select_item_df.head()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:05.642127Z","iopub.execute_input":"2021-05-26T21:27:05.642452Z","iopub.status.idle":"2021-05-26T21:27:05.659628Z","shell.execute_reply.started":"2021-05-26T21:27:05.642423Z","shell.execute_reply":"2021-05-26T21:27:05.658241Z"}}
        select_item_df[(select_item_df.user_id == 'TEsMT2ssaIOeeYroq47ul7Byzub2') & (
            select_item_df.item_name == 'Soul')]

        # %% [markdown]
        # ### droping duplicate rows and unnecessary columns

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:05.661058Z","iopub.execute_input":"2021-05-26T21:27:05.661382Z","iopub.status.idle":"2021-05-26T21:27:05.696986Z","shell.execute_reply.started":"2021-05-26T21:27:05.661348Z","shell.execute_reply":"2021-05-26T21:27:05.695484Z"}}
        item_selection = select_item_df.drop_duplicates(
            subset=['user_id', 'firebase_screen', 'item_name'], keep='first')
        item_selection.drop(['ITEM_POSITION_LIST'],
                            axis='columns', inplace=True)
        item_selection

        # %% [markdown]
        # ### including weight = 10 time number of time an item/product is clicked

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:05.698703Z","iopub.execute_input":"2021-05-26T21:27:05.699003Z","iopub.status.idle":"2021-05-26T21:27:05.728219Z","shell.execute_reply.started":"2021-05-26T21:27:05.698975Z","shell.execute_reply":"2021-05-26T21:27:05.726606Z"}}
        item_selection['unscalled_weights'] = item_selection.total_item_counts * 10
        item_selection.drop(['total_item_counts', 'item_name'],
                            axis='columns', inplace=True)
        item_selection

        # %% [markdown]
        # ### item_selection is our final set of data

        # %% [markdown]
        # * * * * ## Merging all the dfs

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:05.730274Z","iopub.execute_input":"2021-05-26T21:27:05.730851Z","iopub.status.idle":"2021-05-26T21:27:05.764837Z","shell.execute_reply.started":"2021-05-26T21:27:05.730801Z","shell.execute_reply":"2021-05-26T21:27:05.763659Z"}}
        frames = [screen_view, content_selection, item_selection]
        result = pd.concat(frames)
        result

        # calculate ratings now

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:05.766570Z","iopub.execute_input":"2021-05-26T21:27:05.767192Z","iopub.status.idle":"2021-05-26T21:27:05.774881Z","shell.execute_reply.started":"2021-05-26T21:27:05.767143Z","shell.execute_reply":"2021-05-26T21:27:05.773931Z"}}
        result.drop(['event_date'], axis='columns', inplace=True)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:05.776282Z","iopub.execute_input":"2021-05-26T21:27:05.776788Z","iopub.status.idle":"2021-05-26T21:27:06.189832Z","shell.execute_reply.started":"2021-05-26T21:27:05.776744Z","shell.execute_reply":"2021-05-26T21:27:06.188544Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        result.groupby(['user_id', 'firebase_screen']).sum()['unscalled_weights'].unstack().plot(kind='bar', ax=ax, alpha=0.75,
                                                                                                 rot=50, title='sum of unscalled weights for each layout based on item, content selection and engagement time of each user')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.191253Z","iopub.execute_input":"2021-05-26T21:27:06.191700Z","iopub.status.idle":"2021-05-26T21:27:06.457537Z","shell.execute_reply.started":"2021-05-26T21:27:06.191647Z","shell.execute_reply":"2021-05-26T21:27:06.456444Z"}}
        fig, ax = plt.subplots(figsize=(15, 7))
        result.groupby(['user_id', 'event_name']).count()['firebase_screen'].unstack().plot(
            kind='bar', ax=ax, alpha=0.75, rot=50, title='no of times events are triggered by users on different screens')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.459461Z","iopub.execute_input":"2021-05-26T21:27:06.459766Z","iopub.status.idle":"2021-05-26T21:27:06.479429Z","shell.execute_reply.started":"2021-05-26T21:27:06.459736Z","shell.execute_reply":"2021-05-26T21:27:06.478115Z"}}
        result[(result.user_id == 'lws8XqradLMg6flQjxaISKYCgPk1') |
               (result.user_id == 'U7qsQi7Et7cE10FdgUn9pTjNH3b2')]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.481079Z","iopub.execute_input":"2021-05-26T21:27:06.481425Z","iopub.status.idle":"2021-05-26T21:27:06.497849Z","shell.execute_reply.started":"2021-05-26T21:27:06.481392Z","shell.execute_reply":"2021-05-26T21:27:06.496551Z"}}
        result.describe()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.499339Z","iopub.execute_input":"2021-05-26T21:27:06.499636Z","iopub.status.idle":"2021-05-26T21:27:06.525261Z","shell.execute_reply.started":"2021-05-26T21:27:06.499609Z","shell.execute_reply":"2021-05-26T21:27:06.524412Z"}}
        result['weights'] = result.groupby(['user_id', 'firebase_screen'])[
            'unscalled_weights'].transform('sum')
        result

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.526726Z","iopub.execute_input":"2021-05-26T21:27:06.527010Z","iopub.status.idle":"2021-05-26T21:27:06.552983Z","shell.execute_reply.started":"2021-05-26T21:27:06.526982Z","shell.execute_reply":"2021-05-26T21:27:06.551563Z"}}
        result.describe()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.554206Z","iopub.execute_input":"2021-05-26T21:27:06.554546Z","iopub.status.idle":"2021-05-26T21:27:06.576089Z","shell.execute_reply.started":"2021-05-26T21:27:06.554515Z","shell.execute_reply":"2021-05-26T21:27:06.575263Z"}}
        result = result.drop_duplicates(
            subset=['user_id', 'firebase_screen'], keep='first')
        result.drop(['event_name', 'unscalled_weights'],
                    axis='columns', inplace=True)
        result.head(2)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.577430Z","iopub.execute_input":"2021-05-26T21:27:06.577745Z","iopub.status.idle":"2021-05-26T21:27:06.595102Z","shell.execute_reply.started":"2021-05-26T21:27:06.577714Z","shell.execute_reply":"2021-05-26T21:27:06.593544Z"}}
        result.describe()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.597061Z","iopub.execute_input":"2021-05-26T21:27:06.597669Z","iopub.status.idle":"2021-05-26T21:27:06.620517Z","shell.execute_reply.started":"2021-05-26T21:27:06.597619Z","shell.execute_reply":"2021-05-26T21:27:06.619426Z"}}
        result['scaled_ratings'] = (
            result['weights'] - result['weights'].mean()) / result['weights'].std()
        result.head(2)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.621939Z","iopub.execute_input":"2021-05-26T21:27:06.622371Z","iopub.status.idle":"2021-05-26T21:27:06.640986Z","shell.execute_reply.started":"2021-05-26T21:27:06.622298Z","shell.execute_reply":"2021-05-26T21:27:06.639162Z"}}
        result['scaled_ratings_2'] = (result['weights']-result['weights'].min()) * \
            5/((result['weights'].max())-(result['weights'].min()))
        result.head(2)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.642536Z","iopub.execute_input":"2021-05-26T21:27:06.642893Z","iopub.status.idle":"2021-05-26T21:27:06.668465Z","shell.execute_reply.started":"2021-05-26T21:27:06.642854Z","shell.execute_reply":"2021-05-26T21:27:06.667658Z"}}
        # implicit ratings normalization
        result['scaled_ratings_3'] = 5 * \
            result['weights'] / result['weights'].max()
        result

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.669637Z","iopub.execute_input":"2021-05-26T21:27:06.670088Z","iopub.status.idle":"2021-05-26T21:27:06.673449Z","shell.execute_reply.started":"2021-05-26T21:27:06.670043Z","shell.execute_reply":"2021-05-26T21:27:06.672671Z"}}
        #result['scaled_ratings_rounded'] = round(result['scaled_ratings_3'])
        # result

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.674641Z","iopub.execute_input":"2021-05-26T21:27:06.675119Z","iopub.status.idle":"2021-05-26T21:27:06.826242Z","shell.execute_reply.started":"2021-05-26T21:27:06.675088Z","shell.execute_reply":"2021-05-26T21:27:06.825528Z"}}
        plt.plot(result['weights'], result['scaled_ratings'])
        plt.show()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.827293Z","iopub.execute_input":"2021-05-26T21:27:06.827742Z","iopub.status.idle":"2021-05-26T21:27:06.950860Z","shell.execute_reply.started":"2021-05-26T21:27:06.827711Z","shell.execute_reply":"2021-05-26T21:27:06.949585Z"}}
        plt.plot(result['scaled_ratings_2'], result['scaled_ratings'])
        plt.show()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:06.952175Z","iopub.execute_input":"2021-05-26T21:27:06.952487Z","iopub.status.idle":"2021-05-26T21:27:07.076913Z","shell.execute_reply.started":"2021-05-26T21:27:06.952450Z","shell.execute_reply":"2021-05-26T21:27:07.075874Z"}}
        plt.plot(result['scaled_ratings_3'], result['scaled_ratings'])
        plt.show()

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.078079Z","iopub.execute_input":"2021-05-26T21:27:07.078565Z","iopub.status.idle":"2021-05-26T21:27:07.081609Z","shell.execute_reply.started":"2021-05-26T21:27:07.078529Z","shell.execute_reply":"2021-05-26T21:27:07.080809Z"}}
        #ax = result.plot.scatter(x='scaled_ratings_3', y='scaled_ratings_rounded', c='DarkBlue')

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.082695Z","iopub.execute_input":"2021-05-26T21:27:07.083114Z","iopub.status.idle":"2021-05-26T21:27:07.096791Z","shell.execute_reply.started":"2021-05-26T21:27:07.083083Z","shell.execute_reply":"2021-05-26T21:27:07.095217Z"}}
        #ax1 = plt.bar(x = result['scaled_ratings_3'], y = result['scaled_ratings_rounded'],rot=0)
        #ax = result.plot.bar(x='scaled_ratings_3', y='scaled_ratings_rounded', rot=0)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.098734Z","iopub.execute_input":"2021-05-26T21:27:07.099069Z","iopub.status.idle":"2021-05-26T21:27:07.109665Z","shell.execute_reply.started":"2021-05-26T21:27:07.099031Z","shell.execute_reply":"2021-05-26T21:27:07.108160Z"}}
        # keeping the rating 3 that is calculated according to the siggested method by kim in his book and deleting the rest

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.111496Z","iopub.execute_input":"2021-05-26T21:27:07.111885Z","iopub.status.idle":"2021-05-26T21:27:07.125926Z","shell.execute_reply.started":"2021-05-26T21:27:07.111843Z","shell.execute_reply":"2021-05-26T21:27:07.124700Z"}}
        result.drop(['weights', 'scaled_ratings', 'scaled_ratings_2'],
                    axis='columns', inplace=True)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.128412Z","iopub.execute_input":"2021-05-26T21:27:07.128995Z","iopub.status.idle":"2021-05-26T21:27:07.145632Z","shell.execute_reply.started":"2021-05-26T21:27:07.128917Z","shell.execute_reply":"2021-05-26T21:27:07.144317Z"}}
        # renaming the columns as "user_id" to "users", "firebase_screen" to "layouts" and  "scaled_ratings_rounded" to "ratings"
        result = result.rename(
            columns={"scaled_ratings_3": "ratings", "firebase_screen": "layout_id"})
        result.reset_index(inplace=True)
        result.loc[(result.ratings > 0) & (result.ratings < 1), 'ratings'] = 1

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.164611Z","iopub.execute_input":"2021-05-26T21:27:07.164972Z","iopub.status.idle":"2021-05-26T21:27:07.186122Z","shell.execute_reply.started":"2021-05-26T21:27:07.164941Z","shell.execute_reply":"2021-05-26T21:27:07.184236Z"}}
        result.ratings = round(result.ratings, 0)
        result = result.drop(['index'], axis=1)
        result

        # %% [markdown]
        # ## pivot table for sprasity checking

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.192555Z","iopub.execute_input":"2021-05-26T21:27:07.192982Z","iopub.status.idle":"2021-05-26T21:27:07.207537Z","shell.execute_reply.started":"2021-05-26T21:27:07.192937Z","shell.execute_reply":"2021-05-26T21:27:07.205774Z"}}
        Ratings = result.pivot(
            index='user_id',
            columns='layout_id',
            values='ratings'
        )
        Ratings = Ratings.fillna(0).values
        print('type ', type(Ratings))
        print('shape ', Ratings.shape)
        print('Ratings ', Ratings)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.209256Z","iopub.execute_input":"2021-05-26T21:27:07.209708Z","iopub.status.idle":"2021-05-26T21:27:07.219235Z","shell.execute_reply.started":"2021-05-26T21:27:07.209661Z","shell.execute_reply":"2021-05-26T21:27:07.218197Z"}}
        sparsity = float(len(Ratings.nonzero()[0]))
        sparsity /= (Ratings.shape[0] * Ratings.shape[1])
        sparsity *= 100
        print('{:.2f}%'.format(sparsity))

        # %% [markdown]
        # ## in order to recommend the layouts/items to users first task is to convert the sparse matrix i.e., 70% to a dense matrix

        # %% [markdown]
        # ### in order to do that our approach is to adopt the matrix factorization approach

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.221109Z","iopub.execute_input":"2021-05-26T21:27:07.221493Z","iopub.status.idle":"2021-05-26T21:27:07.233710Z","shell.execute_reply.started":"2021-05-26T21:27:07.221456Z","shell.execute_reply":"2021-05-26T21:27:07.232416Z"}}
        # from sklearn.model_selection import train_test_split
        # # dividing the dataset into the traing and validation sets

        # def split_data(ratings):

        #     training_df, validation_df = train_test_split(Ratings, test_size=.2, random_state=42)
        #     return  training_df, validation_df
        # trainsplit_data(result)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.235133Z","iopub.execute_input":"2021-05-26T21:27:07.235474Z","iopub.status.idle":"2021-05-26T21:27:07.247078Z","shell.execute_reply.started":"2021-05-26T21:27:07.235444Z","shell.execute_reply":"2021-05-26T21:27:07.246269Z"}}
        Ratings.shape

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.248123Z","iopub.execute_input":"2021-05-26T21:27:07.248634Z","iopub.status.idle":"2021-05-26T21:27:07.261322Z","shell.execute_reply.started":"2021-05-26T21:27:07.248602Z","shell.execute_reply":"2021-05-26T21:27:07.260087Z"}}
        # training_df.shape

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.262935Z","iopub.execute_input":"2021-05-26T21:27:07.263254Z","iopub.status.idle":"2021-05-26T21:27:07.273970Z","shell.execute_reply.started":"2021-05-26T21:27:07.263222Z","shell.execute_reply":"2021-05-26T21:27:07.272776Z"}}
        # np.size(training_df,0)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.275299Z","iopub.execute_input":"2021-05-26T21:27:07.275826Z","iopub.status.idle":"2021-05-26T21:27:07.290743Z","shell.execute_reply.started":"2021-05-26T21:27:07.275783Z","shell.execute_reply":"2021-05-26T21:27:07.289764Z"}}
        # training_df[:,1]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.292080Z","iopub.execute_input":"2021-05-26T21:27:07.292670Z","iopub.status.idle":"2021-05-26T21:27:07.303238Z","shell.execute_reply.started":"2021-05-26T21:27:07.292622Z","shell.execute_reply":"2021-05-26T21:27:07.302381Z"}}
        # training_df[:,0]

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.304377Z","iopub.execute_input":"2021-05-26T21:27:07.304810Z","iopub.status.idle":"2021-05-26T21:27:07.315112Z","shell.execute_reply.started":"2021-05-26T21:27:07.304779Z","shell.execute_reply":"2021-05-26T21:27:07.314097Z"}}
        # type(training_df)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.316219Z","iopub.execute_input":"2021-05-26T21:27:07.316710Z","iopub.status.idle":"2021-05-26T21:27:07.328843Z","shell.execute_reply.started":"2021-05-26T21:27:07.316679Z","shell.execute_reply":"2021-05-26T21:27:07.327960Z"}}

        #from decimal import Decimal

        # dividing the dataset into the traing and validation sets

        # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
        # logger = logging.getLogger()
        # logger.setLevel(logging.DEBUG)

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.329973Z","iopub.execute_input":"2021-05-26T21:27:07.330478Z","iopub.status.idle":"2021-05-26T21:27:07.345744Z","shell.execute_reply.started":"2021-05-26T21:27:07.330445Z","shell.execute_reply":"2021-05-26T21:27:07.344716Z"}}
        logging.debug("debug")

        logging.info("info")

        logging.warning("warning")

        logging.error("error")
        print("y")

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.347369Z","iopub.execute_input":"2021-05-26T21:27:07.347684Z","iopub.status.idle":"2021-05-26T21:27:07.358165Z","shell.execute_reply.started":"2021-05-26T21:27:07.347644Z","shell.execute_reply":"2021-05-26T21:27:07.356679Z"}}

        # Create and configure logger
        logging.basicConfig(filename="newfile.log",
                            format='%(asctime)s %(message)s',
                            filemode='w')

        # Creating an object
        logger = logging.getLogger()

        # Setting the threshold of logger to DEBUG
        logger.setLevel(logging.DEBUG)

        # Test messages
        logger.debug("Harmless debug Message")
        logger.info("Just an information")
        logger.warning("Its a Warning")
        logger.error("Did you try to divide by zero")
        logger.critical("Internet is down")

        # %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T21:27:07.360113Z","iopub.execute_input":"2021-05-26T21:27:07.360596Z","iopub.status.idle":"2021-05-26T21:27:08.048274Z","shell.execute_reply.started":"2021-05-26T21:27:07.360546Z","shell.execute_reply":"2021-05-26T21:27:08.046908Z"}}
        return result
