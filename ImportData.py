from oauth2client.client import GoogleCredentials
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from google.oauth2 import service_account
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
import seaborn as sns
from google.cloud import bigquery
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'etbd-1-2c366b9f240d.json'


class ImportData:

    client = bigquery.Client()
    gcp_project = 'etbd-1'

    # def __init__(self):
    #     # self.import_bq_data()
    #     self.dataset = None

    #     self.URL1 = 'https://drive.google.com/file/d/1R2whoZWVpLIZpaZpL9hzg89iDmy0pamp/view?usp=sharing'
    #     self.path1 = 'https://drive.google.com/uc?export=download&id=' + \
    #         self.URL1.split('/')[-2]
    #     self.df1 = pd.read_csv(self.path1)

    #     self.URL2 = 'https://drive.google.com/file/d/11GJBM9hvJzHz8GLcefs8fflghtKItiBi/view?usp=sharing'
    #     self.path2 = 'https://drive.google.com/uc?export=download&id=' + \
    #         self.URL2.split('/')[-2]
    #     self.df2 = pd.read_csv(self.path2)

    #     self.URL3 = 'https://drive.google.com/file/d/10of_7LOzuGpPkgv4hHhliHhNPY8jZFEx/view?usp=sharing'
    #     self.path3 = 'https://drive.google.com/uc?export=download&id=' + \
    #         self.URL3.split('/')[-2]
    #     self.df3 = pd.read_csv(self.path3)

    #     self.URL4 = 'https://drive.google.com/file/d/1mJgAzNs01iX4wAk7hKSRgUZp4j2K9faR/view?usp=sharing'
    #     self.path4 = 'https://drive.google.com/uc?export=download&id=' + \
    #         self.URL4.split('/')[-2]
    #     self.df4 = pd.read_csv(self.path4)

    #     self.URL5 = 'https://drive.google.com/file/d/1z5OMKuRrd8tWTFDA2RkVbBo1e0Kxo7us/view?usp=sharing'
    #     self.path5 = 'https://drive.google.com/uc?export=download&id=' + \
    #         self.URL5.split('/')[-2]
    #     self.df5 = pd.read_csv(self.path5)

    #     self.dataset = self.df5.append(
    #         [self.df1, self.df2, self.df3, self.df4], ignore_index=True)
    #     print("dataset1", type(self.dataset))

    #     print("dataset2", type(self.dataset))
    #     if not self.dataset.empty:
    #         print("dataset3", type(self.dataset))
    #         self.import_bq_data()

    def import_bq_data(self, dataset):
        self.dataset = dataset
        # return self.val
        #self.import_df = self.import_data()
        print("self.val", self.dataset)
        # columns = df5.columns
        self.data_clanning = self.data_clanning(self.dataset)
        print("type(self.df.columns)", type(self.data_clanning))
        return (self.val1.columns)

    def data_clanning(self, df):
        self.data_clanning_df = df
        # self.less_features = self.drop_unnecessary_features(
        #     self.data_clanning_df)
        self.droped_duplicate_timestamps = self.find_duplicate_timestamp(
            self.data_clanning_df)
        # self.selected_dates = self.event_date(self.droped_duplicate_timestamps)
        # self.removedunnecessary_rows = self.removing_unnecessary_rows(
        #     self.selected_dates)
        # self.engagement_time_formating = self.formating_engagement_time(
        #     self.removedunnecessary_rows)

        return self.droped_duplicate_timestamps
        # return self.removedunnecessary_rows

        #print("self.df_drop_features", self.df_drop_features)
        return self.df_drop_features

    def find_duplicate_timestamp(self, df):
        self.event_timestamp = df
        print("self.event_timestamp", self.event_timestamp)
        print('duplicate timestamp: {} ' .format(self.df.loc[:, [
              'user_id', 'firebase_screen', 'event_date', 'event_name', 'event_timestamp']]))
        # drop duplicates
        # self.df.drop_duplicates(subset=['event_timestamp'])
        return self.df

    def event_date(self, df):
        self.df = df
        print("df.event_date.unique(): {}".format(
            self.df.event_date.unique()))
        # get only the dates with new layouts items and design
        self.df = self.df.loc[(self.df.event_date >= '2021-02-03'), :]
        self.df.info()
        print('data since 2021-02-03 ' .format(
            [self.df.loc[:, ['user_id', 'firebase_screen', 'event_date', 'event_name', 'item_name', 'content']]]))
        print("Describe Dataframe: {}", format(self.df.describe()))
        return self.df

    def removing_unnecessary_rows(self, df):
        # find if any event_name, firebase_screen/layouts is null or unnecessary values existing
        self.df = df
        self.nonNull_firebase_screen = self.df[(self.df.firebase_screen.notnull()) & (
            self.df.event_name.notnull()) & (self.df.event_name != 'my_tag_event')]
        print("null or unncessary values of firebase_screen or firebase_screen exist : {}/n  " .format(
            [self.nonNull_firebase_screen.loc[:, ['firebase_screen', 'event_name']]]))
        return self.nonNull_firebase_screen

    def formating_engagement_time(self, df):
        self.df = df
        # converting time to seconds from m-seconds
        self.df['engagement_time_seconds'] = self.df.engagement_time_msec/1000
        self.df = self.df.drop(['engagement_time_msec'], axis=1)
        self.df = self.df[(~self.df.engagement_time_seconds.isna())]
        print("engagement_time_seconds final", self.df.head())
        return self.df
