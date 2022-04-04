import constants as const


def embedding_matrix_pkl_file_name(name):
    return f'{const.embedding_path}{name}_embedding_matrix.npy'


def balanced_set(dataset_type, balanced_name):
    return f'{const.data_path}balanced_{dataset_type}_{balanced_name}.csv'


def balanced_subset(dataset_type, balanced_name, set_type):
    return f'{const.data_path}balanced_{dataset_type}_{balanced_name}_{set_type}.csv'


def dataset(name):
    return f'{const.yelp_path}{name}.json'


def pickled_tokens(set, task):
    return f'{const.pickled_path}{task}_{set}_tokens.pkl'


def pickled_dataset(type):
    return f'{const.pickled_path}{type}.pkl'


def picked_cleaned_sentences(set, task):
    return f'{const.pickled_path}{task}_{set}_cleaned_sentences.pkl'
