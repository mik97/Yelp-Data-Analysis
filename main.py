# execution times
# File loaded and pickled in 7.5567114869753516 seconds
# File unpickled in 1.1141046166419983 minutes

import os
import dataset_handler as dh
from os import path
from time import time
import pandas as pd
import preprocessing


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
    balanced_df = dh.get_balanced_subset(data, 'sentiment')
    
    # pipeline
    test_data = balanced_df.head(100)

    df = pd.read_csv("./test.csv")
    print(df)
    start_time = time()

    texts = preprocessing.to_lower(df)
    print(texts)
    texts = texts.apply(preprocessing.decontract)
    print(texts)
    tokens = preprocessing.get_tokens(texts)
    print("%s seconds" % (time() - start_time))
    # print(tokens[1])
    # lowerTokens = list(map(preprocessing.toLowerCase, tokens))
    # print(lowerTokens[1])
    start_time = time()
    stop_words = preprocessing.remove_stopwords_and_noalpha(tokens)
    print("%s seconds" % (time() - start_time))
    print(stop_words)
    lemm = preprocessing.to_lemmas(stop_words)
    print(lemm)


if __name__ == "__main__":
    main()
