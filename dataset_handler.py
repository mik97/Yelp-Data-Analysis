import os
import time
import const

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


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

    return df_toRet


def analyze(df, type):
    #  create the folder where to save the plots
    # checkPlotFolder()

    if (type == 'review'):
        checkPlotFolder('review')
        savePath = const.plotsPath + '/review/'

        # how many sample for each stars rating
        starsCounted = df["stars"].value_counts().sort_index()

        plt.bar(['1', '2', '3', '4', '5'], starsCounted)
        plt.title("Review ratings count")

        saveFigure(savePath + "reviewRatingsCount.jpg")

        # how many positive or negative samples
        sentValues = df['sentiment'].value_counts().sort_index()

        plt.bar(['negative', 'positive'], sentValues, width=0.5)
        plt.title("Positive and negative sentiment count")

        saveFigure(savePath + "posNegSentimentCount.jpg")


def checkPlotFolder(type=None):
    directory_to_create = const.plotsPath

    if (type):
        directory_to_create += '/%s' % type

    if (not os.path.exists(directory_to_create)):
        try:
            os.makedirs(directory_to_create)
        except OSError:
            print("Creation of the directory %s failed" % directory_to_create)
        else:
            print("Successfully created of the directory %s" %
                  directory_to_create)


def saveFigure(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
        plt.savefig(filepath)
    except FileNotFoundError:
        print('%s not found' % filepath)
    plt.clf()


def get_balanced_subset(data, col):
    if(col == 'sentiment'):
        negSamples = data[data[col] == 0.0].sample(n=500000)
        posSamples = data[data[col] == 1.0].sample(n=500000)

        final_df = shuffle(
            pd.concat([negSamples, posSamples])).reset_index(drop=True)
        print(final_df)
        return final_df
