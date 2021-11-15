import const
import time


def pickled_file_name(name):
    return f'{const.pkl_path}{name}.pkl'


def balanced_data_file_name(dataset_type, balanced_name):
    return f'{const.balanced_csv_path}{dataset_type}_{balanced_name}.csv'


def plot_file_name(name):
    return f'{const.plots_path}{name}.png'


def get_tokens_file(task_name):
    return f'{const.pkl_path}tokens_{task_name}.pkl'


def get_w2v_file(task_name):
    return f'{const.w2v_path}w2v_{task_name}.bin'


def get_minutes(start_time):
    return (time.time() - start_time)/60
