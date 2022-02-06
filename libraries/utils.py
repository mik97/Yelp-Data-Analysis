import constants as const

import pickle
import time

# get paths


def pickled_file_name(name):
    return f'{const.pickled_path}{name}.pkl'


def embedding_matrix_pkl_file_name(name):
    return f'{const.embedding_path}{name}_embedding_matrix.npy'


def balanced_data_file_name(dataset_type, balanced_name, set_type):
    return f'{const.data_path}balanced_{dataset_type}_{balanced_name}_{set_type}.csv'


def dataset_file_name(name):
    return f'{const.yelp_path}{name}.json'


def tokens_file_name(set, task):
    return f'{const.pickled_path}{task}_{set}_tokens.pkl'


def cleaned_sentences_file_name(set, task):
    return f'{const.pickled_path}{task}_{set}_cleaned_sentences.pkl'


def get_minutes(start_time):
    return round(((time.time() - start_time)/60), 2)


def save_pickled(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickled(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
