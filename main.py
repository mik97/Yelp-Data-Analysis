import os
import const
import preprocessing

import dataset_handler as dh


def main():
    review_dataset_tasks()


def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''
    review_df = None

    dh.checkDatasetBalancedFolder()
    # check existence of balanced dataset versions, if not, create them
    balanced_data_files = ['./balanced_dataset/dataset_sentiment.csv']

    if not all([os.path.exists(file) for file in balanced_data_files]):
        review_df = dh.initDataset('review')
        dh.analyze(review_df, 'review')

    task1_pipeline(review_df)
    # task2_pipeline(review_df)


def task1_pipeline(data):
    cleaned_sentences = None

    if not os.path.exists("./data/tokens_task1.pkl"):
        # get a dataset sentiment balanced, if it already exists load from csv
        balanced_df = dh.get_balanced_subset(data, 'sentiment')
        # pipeline
        # pass only the series with text
        cleaned_sentences = preprocessing.preprocess_text(
            balanced_df['text'], name='task1')
    else:
        cleaned_sentences = preprocessing.load_preprocessed_text(name='task1')

    dh.checkW2vModelsFolder()

    # get word embedding of reviews
    w2vec_model = preprocessing.get_word_embedding(
        cleaned_sentences, name='task1')

    print(w2vec_model.wv['burrito'])


if __name__ == "__main__":
    main()
