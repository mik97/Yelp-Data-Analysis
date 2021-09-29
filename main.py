# execution times
# File loaded and pickled in 7.5567114869753516 seconds
# File unpickled in 1.1141046166419983 minutes
import os
import const

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

    if not all([os.path.exists(file) for file in balanced_data_files]):
        review_df = dh.initDataset('review')
        dh.analyze(review_df, 'review')

    task1_pipeline(review_df)
    # task2_pipeline(review_df)


def task1_pipeline(data):
    cleaned_sentences = None

    if const.do_preprocessing:
        # get a dataset sentiment balanced, if it already exists load from csv
        balanced_df = dh.get_balanced_subset(data, 'sentiment')
        # pipeline
        # pass only the series with text
        cleaned_sentences = preprocessing.text_pipeline(balanced_df['text'])
    else:
         cleaned_sentences = preprocessing.text_pipeline()
    
    print(cleaned_sentences[0])
    print(len(cleaned_sentences))

    # get word embedding of reviews
    w2vec_model = preprocessing.get_word_embedding(cleaned_sentences)
    print(w2vec_model.wv['burrito'])


if __name__ == "__main__":
    main()
