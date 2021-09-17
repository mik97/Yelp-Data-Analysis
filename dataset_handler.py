import os
import time

from pandas.core.series import Series
import const

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def initDataset(type):
    '''
        Load the entire dataset and pickle it or read it directly from an existing file containing a pickled dataframe.

        Parameters:
            type (string): "business", "checkin", "review", "tip" or "user".
    '''
    start_time = time.time()

    df_toRet = []
    pkl_filepath = const.pklPath + type + '_data.pkl'

    if os.path.exists(pkl_filepath):
        print("Unpickling existing file . . .")

        df_toRet = pd.read_pickle(pkl_filepath)

        print("File unpickled in {0} minutes".format(
            (time.time() - start_time)/60))
    else:
        print("Loading file . . .")
        total = []
        # remember: each chunk is a regular dataframe object
        # 864 chunks w chunk == 10_000
        index = 1
        for chunk in pd.read_json(const.filesPath[type], lines=True, orient="records", chunksize=10_000):
            if (index % 50 == 0):
                print("\t{0} chunks loaded".format(index))

            if (type == 'review'):
                # add sentiment column
                chunk.loc[chunk['stars'] <= 3, 'sentiment'] = 0
                chunk.loc[chunk['stars'] > 3, 'sentiment'] = 1
            total.append(chunk)

            index = index + 1

        df_toRet = pd.concat(total, ignore_index=True)

        print("File loaded {0} minutes".format((time.time() - start_time)/60))

        # pickle the file if not already picklef
        pd.to_pickle(df_toRet, pkl_filepath)

    print("Loaded file with {0} rows and {1} columns".format(
        df_toRet.shape[0], df_toRet.shape[1]))

    print(df_toRet[['stars', 'sentiment']].head(10))

    return df_toRet


def analyze(df, type):
    if (type == 'review'):
        # how many sample for each stars rating
        starsCounted = df["stars"].value_counts()
        starsCounted = starsCounted.sort_index()
        plt.bar(['1', '2', '3', '4', '5'], starsCounted)
        plt.show()
        # how many positive or negative samples
        pass
