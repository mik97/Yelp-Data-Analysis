# execution times
# File loaded and pickled in 7.5567114869753516 seconds
# File unpickled in 1.1141046166419983 minutes

import os

import pandas as pd
import dataset_handler as dh
import preprocessing

from time import time


def main():
    review_dataset_tasks()
   

def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''

    review_df = None

    # check existence of balanced dataset versions, if not, create them
    balanced_data_files = ['./balanced_dataset/dataset_sentiment.csv']
    
    if not all([os.path.exists(file) for file in balanced_data_files]) :
        review_df = dh.initDataset('review')
        dh.analyze(review_df, 'review')

    task1_pipeline(review_df)
    # task2_pipeline(review_df)


def task1_pipeline(data):
    # get a dataset sentiment balanced, if it already exists load from csv

    # balanced_df = dh.get_balanced_subset(data, 'sentiment')
    
    # pipeline
    df = pd.read_csv("./test.csv")
    
    # pass only the series with text
    stuff = preprocessing.text_pipeline(df['text'])
   


if __name__ == "__main__":
    main()
