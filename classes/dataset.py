
from sklearn.model_selection import train_test_split
import const
import data_handler
import utils
import os

import pandas


class Dataset():
    # based on data analytics slides, a good splitting for huge dataset (>= 1_000_000)
    # is 0.98 train, 0.01 test ad 0.01 val

    def __init__(self, dataset_name, column_to_balance):
        '''
        dataset_name: 'review', 'business'
        column_to_balance: name
        '''
        self.dataset_name = dataset_name
        self.column_to_balance = column_to_balance
        self.subsets_files = {
            'train': utils.data_file_name(self.dataset_name, self.column_to_balance, 'train'),
            'val': utils.data_file_name(self.dataset_name, self.column_to_balance, 'val'),
            'test': utils.data_file_name(self.dataset_name, self.column_to_balance, 'test')

        }

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def split(self, x_columns, target, val_size=0.01, test_size=0.01, n_samples=500_000):
        ''' Load splitted versions or split here'''

        if all([os.path.exists(path) for path in list(self.subsets_files.values())]):
            # load existing files
            train = data_handler.load_subset(self.subsets_files['train'])
            self.train_data = (train[x_columns], train[target])

            val = data_handler.load_subset(self.subsets_files['val'])
            self.val_data = (val[x_columns], val[target])

            test = data_handler.load_subset(self.subsets_files['test'])
            self.test_data = (test[x_columns], test[target])

        else:
            data = data_handler.get_balanced_subset(
                self.dataset_name, self.column_to_balance, n_samples)

            self._split(data[x_columns], data[target], val_size, test_size)

    def _split(self, x_data, y_data, val_size, test_size):

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_size, random_state=const.seed)

        print(x_train, y_train)

        # adjust % validation set
        # tot_samples : 100 = x : val %
        n_val_samples = x_data.shape[0] * val_size
        # train_samples : 100 = val samples : x
        adjusted_val_size = n_val_samples / x_train.shape[0]

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=adjusted_val_size, random_state=const.seed)

        print(
            f'Dataset splitted into train ({x_train.shape[0]} samples), val ({x_val.shape[0]} samples), test ({x_test.shape[0]} samples)')

        self.train_data = (x_train, y_train)
        self.val_data = (x_val, y_val)
        self.test_data = (x_test, y_test)

        # save csv

        x_train.join(y_train).to_csv(self.subsets_files['train'])
        print(f"{self.subsets_files['train']} created")

        x_val.join(y_val).to_csv(self.subsets_files['val'])
        print(f"{self.subsets_files['val']} created")

        x_test.join(y_test).to_csv(self.subsets_files['test'])
        print(f"{self.subsets_files['test']} created")
