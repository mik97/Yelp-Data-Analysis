# In order to speed up the first loading, if the dataset pickled version doesn't exist,
# we load the entire version from original csv and then pickle it.

import time
import os

import const
import utils

import pandas as pd
from sklearn.utils import shuffle


def load_dataset(type):
    '''
    Load the entire dataset from Yelp if balanced version doesn't already exist.

    Parameters:
        dataset type (string): "business", "checkin", "review", "tip" or "user". '''

    path = utils.pickled_file_name(type)

    df_dataset = None

    print(f"Loading {type} dataset...")
    if os.path.exists(path):
        unpickle_file(path)
    else:
        # read directly the csv and the save it as a pickled version
        print(f"\tReading {path}...")
        start_time = time.time()

        total = []
        # remember: each chunk is a regular dataframe object
        # 864 chunks w chunk == 10_000
        for chunk_index, chunk in enumerate(pd.read_json(const.filesPath[type], lines=True, orient="records", chunksize=10_000)):
            total.append(_handle_chunk(chunk, type))

            if (chunk_index % 50 == 0):
                print(f"\t\t{chunk_index} chunks loaded")

        df_dataset = pd.concat(total, ignore_index=True)

        print(f"\tFile loaded in {utils.get_minutes(start_time)} minutes")
        pd.to_pickle(df_dataset, path)

    print("Loaded dataset with {0} rows and {1} columns".format(
        df_dataset.shape[0], df_dataset.shape[1]))

    return df_dataset


def unpickle_file(path):
    start_time = time.time()

    print(f"\tUnpickling{path}...")
    unpickled_file = pd.read_pickle(path)
    print(f"\tFile unpickled in {utils.get_minutes(start_time)} minutes")

    return unpickled_file


def _handle_chunk(chunk, type):
    ''' i.e for review dataset add sentiment column'''

    if (type == 'review'):
        chunk.loc[chunk['stars'] <= 3, 'sentiment'] = 0
        chunk.loc[chunk['stars'] > 3, 'sentiment'] = 1

    return chunk


# 500_000
def get_balanced_subset(dataset_name, column_to_balance, n_samples):
    ''' 
        dataset_name: 'review', 'tips' ...
        column_to_balance: name of the column to balance,
        n_samples: samples to get for each balance
    '''
    csv_filepath = utils.balanced_data_file_name(
        dataset_name, column_to_balance)

    balanced_df = None

    print(
        f'Getting {dataset_name} data balanced respect {column_to_balance} values')

    if (os.path.exists(csv_filepath)):
        print(f"\tReading {csv_filepath}...")
        balanced_df = pd.read_csv(csv_filepath)
    else:
        print(
            f"\tBalanced dataset doesn't already exists, generating {csv_filepath}")

        # retrieve data and balance it
        balanced_df = _balance_data(load_dataset(dataset_name),
                                    dataset_name, column_to_balance, n_samples)

        balanced_df.to_csv(csv_filepath)

    return balanced_df


def _balance_data(data, dataset_name, column_to_balance, n_samples):
    ''' Return balanced data base on divverent combinations'''
    to_ret = None

    if dataset_name == 'review':
        if column_to_balance == 'sentiment':
            negSamples = data.loc[data['sentiment'] == 0].sample(n_samples)
            posSamples = data.loc[data['sentiment'] == 1].sample(n_samples)

            to_ret = shuffle(
                pd.concat([negSamples, posSamples]), random_state=const.seed).reset_index()

    return to_ret


# def analyze(df, type):
#     #  create the folder where to save the plots
#     # checkPlotFolder()

#     if (type == 'review'):
#         checkPlotFolder('review')
#         savePath = const.plots_path + '/review/'

#         # how many sample for each stars rating
#         starsCounted = df["stars"].value_counts().sort_index()

#         plt.bar(['1', '2', '3', '4', '5'], starsCounted)
#         plt.title("Review ratings count")

#         saveFigure(savePath + "reviewRatingsCount.jpg")

#         # how many positive or negative samples
#         sentValues = df['sentiment'].value_counts().sort_index()

#         plt.bar(['negative', 'positive'], sentValues, width=0.5)
#         plt.title("Positive and negative sentiment count")

#         saveFigure(savePath + "posNegSentimentCount.jpg")


# def checkPlotFolder(type=None):
#     directory_to_create = const.plots_path

#     if (type):
#         directory_to_create += '/%s' % type

#     if (not os.path.exists(directory_to_create)):
#         try:
#             os.makedirs(directory_to_create)
#         except OSError:
#             print("Creation of the directory %s failed" % directory_to_create)
#         else:
#             print("Successfully created of the directory %s" %
#                   directory_to_create)


# def checkDataFolder():
#     if not os.path.exists('./data'):
#         try:
#             os.makedirs('./data')
#         except:
#             print('Creation of th directory ./data failed')
#         else:
#             print('Successfully created of the directory ./data')


# def checkDatasetBalancedFolder():
#     if not os.path.exists('./balanced_dataset'):
#         try:
#             os.makedirs('./balanced_dataset')
#         except:
#             print('Creation of th directory ./balanced_dataset failed')
#         else:
#             print('Successfully created of the directory ./balanced_dataset')

# def checkW2vModelsFolder():
#     if not os.path.exists('./w2v_models'):
#         try:
#             os.makedirs('./w2v_models')
#         except:
#             print('Creation of th directory ./w2v_models failed')
#         else:
#             print('Successfully created of the directory ./w2v_models')


# def saveFigure(filepath):
#     try:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#         plt.savefig(filepath)
#     except FileNotFoundError:
#         print('%s not found' % filepath)
#     plt.clf()
