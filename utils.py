
import pickle
import const
import time


def pickled_file_name(name):
    return f'{const.pkl_path}{name}.pkl'


def word_embedding_matrix_file_name(name, type):
    return f'{const.embedding_path}{name}.{type}'


def pickled_embedding_matrix_file_name(name):
    return f'{const.embedding_matrix_path}{name}_embedding_matrix.npy'


def balanced_data_file_name(dataset_type, balanced_name):
    return f'{const.data_csv_path}balanced_{dataset_type}_{balanced_name}.csv'


def data_file_name(dataset_name, balanced_name, type):
    return f'{const.data_csv_path}balanced_{dataset_name}_{balanced_name}_{type}.csv'


def dataset_file_name(name):
    return f'{const.dataset_path}{name}.json'


def plot_file_name(name):
    return f'{const.plots_path}{name}.png'


def get_tokens_file(set, task):
    return f'{const.pkl_path}{task}_{set}_tokens.pkl'


def get_cleaned_sen_file(set, task):
    return f'{const.pkl_path}{task}_{set}_cleaned_sentences.pkl'


def get_w2v_file(task_name):
    return f'{const.w2v_path}w2v_{task_name}.bin'


def get_minutes(start_time):
    return round(((time.time() - start_time)/60), 2)


def save_pickled(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickled(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
