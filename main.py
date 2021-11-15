import os
import utils
import preprocessing

import data_handler as data_handler


def main():
    review_dataset_tasks()


def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''

    task1_pipeline()


def task1_pipeline():
    cleaned_sentences = None

    tokens_path = utils.get_tokens_file('task1')
    w2v_path = utils.get_w2v_file('task1')

    # 1. Get preprocessed text
    if os.path.exists(tokens_path):
        cleaned_sentences = preprocessing.load_preprocessed_text(tokens_path)
    else:
        # get review dataset sentiment balanced, if it already exists load from csv
        balanced_df = data_handler.get_balanced_subset(
            'review', 'sentiment', 500_000)

        cleaned_sentences = preprocessing.preprocess_text(
            balanced_df['text'], tokens_path)

    # 2. Get reviews word embedding
    w2vec_model = preprocessing.get_word_embedding(
        cleaned_sentences, w2v_path)

    print(w2vec_model.wv['burrito'])


if __name__ == "__main__":
    main()
