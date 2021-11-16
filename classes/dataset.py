from numpy import newaxis
import const
from sklearn.model_selection import train_test_split


class Dataset():
    # based on data analytics slides, a good splitting for huge dataset (>= 1_000_000)
    # is 0.98 train, 0.01 test ad 0.01 val
    def load_split(self, x_data, y_data, val_size=0.01, test_size=0.01):

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_size, random_state=const.seed)

        # adjust % validation set -----
        # 1. compute the amount of samples for validation set based on total set
        n_val_samples = (x_data.shape[0] * val_size) / 100
        # 2. compute adjusted val % respect train set size
        adjusted_val_size = (100 * n_val_samples) / x_train.shape[0]
        # ------------------

        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data, test_size=adjusted_val_size, random_state=const.seed)

        self.train_data = (x_train, y_train)
        self.val_data = (x_val, y_val)
        self.test_data = (x_test, y_test)

        print(
            f'Dataset splitted into train({x_train.shape[0]} samples), val({x_val.shape[0]} samples), test({x_test.shape[0]} samples)')