import numpy as np
import pandas as pd
import requests
from io import StringIO
from urllib.parse import urljoin
#from tabulate import tabulate


class DataImporting:
    def __init__(self):
        self.dfs = pd.DataFrame()
        self.ds = pd.DataFrame()
        self.list_ll = []
        self.csv_dictionary = {}
        self.file_id_dict = {
            'id1': '1R2whoZWVpLIZpaZpL9hzg89iDmy0pamp/',
            'id2': '11GJBM9hvJzHz8GLcefs8fflghtKItiBi/',
            'id3': '10of_7LOzuGpPkgv4hHhliHhNPY8jZFEx/',
            'id4': '1mJgAzNs01iX4wAk7hKSRgUZp4j2K9faR/',
            'id5': '1z5OMKuRrd8tWTFDA2RkVbBo1e0Kxo7us/',
            'id6': '1z5OMKuRrd8tWTFDA2RkVbBo1e0Kxo7us/'
        }

    def csv_data_importing(self):

        # urlDict = {"orig_url": 'https://drive.google.com/file/d/1R2whoZWVpLIZpaZpL9hzg89iDmy0pamp/view?usp=sharing',
        #            "orig_url": 'https://drive.google.com/file/d/11GJBM9hvJzHz8GLcefs8fflghtKItiBi/view?usp=sharing',
        #            "orig_url": 'https://drive.google.com/file/d/1R2whoZWVpLIZpaZpL9hzg89iDmy0pamp/view?usp=sharing',
        #            "orig_url": 'https://drive.google.com/file/d/1R2whoZWVpLIZpaZpL9hzg89iDmy0pamp/view?usp=sharing',
        #            "orig_url": 'https://drive.google.com/file/d/1R2whoZWVpLIZpaZpL9hzg89iDmy0pamp/view?usp=sharing',
        #            "orig_url": 'https://drive.google.com/file/d/1R2whoZWVpLIZpaZpL9hzg89iDmy0pamp/view?usp=sharing'}

        for key in self.file_id_dict:
            orig_url = 'https://drive.google.com/file/d/' +\
                self.file_id_dict[key]+'view?usp = sharing'
            print("self.orig_url", orig_url)
            file_id = orig_url.split('/')[-2]
            dwn_url = 'https://drive.google.com/uc?export=download&id=' + \
                file_id
            url = requests.get(dwn_url).text

            #print("url", url)
            csv_raw = StringIO(url)
            #print("csv_raw", csv_raw)
            df = pd.read_csv(csv_raw)
            #print('key', key)
            #print('df', df)
            if self.dfs.empty:
                #self.ds = df
                self.dfs = df
                #print("no append", self.dfs)
            else:
                self.dfs = self.dfs.append(df)
                #print("append", self.dfs)
            #print(tabulate(df, headers='keys', tablefmt='psql'))
            #self.csv_dictionary[key] = df

            #print("csv_dictionary", self.csv_dictionary[key])
        # for key, items in self.csv_dictionary.items():
        #     print("values", type(items))
        #     if len(self.ds) > 0:
        #         self.ds.update(key)
        #         print("not self.ds.empty", self.ds)
        #     else:
        #         self.ds = self.ds.update(key)
        #         print("else", self.ds)
        print('type(self.dfs)', type(self.dfs))
        print('len(self.dfs)', len(self.dfs))
        print('self.dfs', self.dfs)
        return self.dfs
