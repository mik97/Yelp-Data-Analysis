# In order to speed up the first loading, if the dataset pickled version doesn't exist,
# we load the entire version from original csv and then pickle it.

from sklearn.utils import shuffle

import pandas as pd

import libraries.utils as utils
import constants as const
import time
import os


def load_dataset(type):
    '''
    Load the entire dataset from Yelp if balanced version doesn't already exist.

    Params:
        dataset type (string): "business", "checkin", "review", "tip" or "user". '''

    print(f"Loading {type} dataset...")

    df_dataset = None

    pickled_data_path = utils.pickled_file_name(type)

    if os.path.exists(pickled_data_path):
        # read pickled version
        df_dataset = unpickle_file(pickled_data_path)
    else:
        # read directly the json and the save it as a pickled version
        data_path = utils.dataset_file_name(type)

        print(f"\tReading {data_path}...")
        start_time = time.time()

        total = []
        # remember: each chunk is a regular dataframe object
        # 864 chunks w chunk == 10_000
        for chunk_index, chunk in enumerate(pd.read_json(data_path, lines=True, orient="records", chunksize=10_000)):
            total.append(_handle_chunk(chunk, type))

            if (chunk_index % 50 == 0):
                print(f"\t\t{chunk_index} chunks loaded")

        df_dataset = pd.concat(total, ignore_index=True)

        print(f"\tFile loaded in {utils.get_minutes(start_time)} minutes")

        pd.to_pickle(df_dataset, pickled_data_path)

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


def get_balanced_subset(dataset_name, column_to_balance, n_samples):
    ''' 
        dataset_name: 'review', 'tips' ...
        column_to_balance: name of the column to balance,
        n_samples: samples to get for each balance
    '''
    # 500_000 samples
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


def load_subset(path):
    print(f"Reading {path}...")
    start_time = time.time()

    total = []
    # remember: each chunk is a regular dataframe object
    # 864 chunks w chunk == 10_000
    for chunk_index, chunk in enumerate(pd.read_csv(path, chunksize=10_000)):
        total.append(chunk)

        if (chunk_index+1 % 50 == 0):
            print(f"\t{chunk_index} chunks loaded")

    df_dataset = pd.concat(total, ignore_index=True)

    print(f"File loaded in {utils.get_minutes(start_time)} minutes")

    return df_dataset