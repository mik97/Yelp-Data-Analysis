import os
from classes.dataset import Dataset
import preprocessing as prep_utils
import task_pipelines as tp
import const
import utils
import pickle


def main():
    review_dataset_tasks()


def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''
    # TODO some statistics and visualization

    tp.task1_pipeline()


if __name__ == "__main__":
    main()
