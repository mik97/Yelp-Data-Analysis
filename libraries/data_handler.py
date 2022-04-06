# In order to speed up the first loading, if the dataset pickled version doesn't exist,
# we load the entire version from original csv and then pickle it.

from sklearn.utils import shuffle

from libraries import utils

import pandas as pd
import time
import os

import libraries.filenames_generator as filenames
import constants as const


def load_dataset(type):
    '''
    Load the entire dataset from Yelp if balanced version doesn't already exist.

    Params:
        dataset type (string): "business", "checkin", "review", "tip" or "user". '''

    pickled_path = filenames.pickled_dataset(type)

    if os.path.exists(pickled_path):
        return unpickle_file(pickled_path)

    data_path = filenames.dataset(type)

    print(f"Loading {type} dataset...")

    total = []

    # remember: each chunk is a regular dataframe object
    # 864 chunks w chunk == 10_000
    for chunk_index, chunk in enumerate(pd.read_json(data_path, lines=True, orient="records", chunksize=10_000)):
        total.append(_handle_chunk(chunk, type))

        if (chunk_index % 50 == 0):
            print(f"\t\t{chunk_index} chunks loaded")

    df_dataset = pd.concat(total, ignore_index=True)

    df_dataset.to_pickle(pickled_path)

    print("Loaded dataset with {0} rows and {1} columns and saved in {2}".format(
        df_dataset.shape[0], df_dataset.shape[1], pickled_path))

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
        #  add sentiment
        chunk.loc[chunk['stars'] <= 3, 'sentiment'] = 0
        chunk.loc[chunk['stars'] > 3, 'sentiment'] = 1

        #  add usefulness
        chunk.loc[chunk['useful'] <= 5, 'usefulness'] = 'not useful'
        chunk.loc[(chunk['useful'] > 5) & (chunk['useful']
                  <= 25), 'usefulness'] = 'moderately useful'
        chunk.loc[chunk['useful'] > 25, 'usefulness'] = 'extremely useful'

        return chunk

    return chunk


def get_balanced_subset(dataset_name, column_to_balance, n_samples):
    '''
        dataset_name: 'review', 'tips' ...
        column_to_balance: name of the column to balance,
        n_samples: samples to get for each balance
    '''
    csv_filepath = filenames.balanced_set(
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
            s1 = data.loc[data['sentiment'] == 0].sample(n_samples)
            s2 = data.loc[data['sentiment'] == 1].sample(n_samples)

            to_ret = shuffle(
                pd.concat([s1, s2]), random_state=const.seed).reset_index()

        elif column_to_balance == 'usefulness':
            counts = data['usefulness'].value_counts()

            s1 = data.loc[data['usefulness'] == 'not useful'].sample(
                min(n_samples, counts['not useful']))
            s2 = data.loc[data['usefulness'] ==
                          'moderately useful'].sample(min(n_samples, counts['moderately useful']))
            s3 = data.loc[data['usefulness'] ==
                          'extremely useful'].sample(min(n_samples, counts['extremely useful']))
            to_ret = shuffle(
                pd.concat([s1, s2, s3]), random_state=const.seed).reset_index()

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
