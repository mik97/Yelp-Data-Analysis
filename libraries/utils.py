import pickle
import time


def get_minutes(start_time):
    return round(((time.time() - start_time)/60), 4)


def save_pickled(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickled(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
