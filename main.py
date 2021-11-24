from classes.dataset import Dataset
import preprocessing as prep


def main():
    review_dataset_tasks()


def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''
    # TODO some statistics and visualization

    task1_pipeline()


def task1_pipeline():
    # tokens_path = utils.get_tokens_file('task1')
    # w2v_path = utils.get_w2v_file('task1')

    # COMPLETE 1. create tokens training set -> save it csv file (column: index, preprocessed text, sentiment)
    # COMPLETE 2. create tokens validation set -> save it csv file (column: index, preprocessed text, sentiment)
    # COMPLETE 3. create tokens test set -> save it csv file (column: index, preprocessed text, sentiment)

    review_data = Dataset('review', 'sentiment')
    review_data.split(['text'], 'sentiment', n_samples=500_000)

    # memotizzare il tokenizer del trainig ( da riutilizzare sia per il val che il test)
    train_tokens, train_tokenizer = prep.preprocess_text(
        review_data.train_data[0])

    val_tokens = prep.preprocess_text(review_data.val_data[0])
    test_tokens = prep.preprocess_text(review_data.test_data[0])

    # use the training tokenizer
    embedding_matrix = prep.get_embedding_matrix(
        train_tokenizer, _, train_tokens)

    # create models that shares tokens and embedding matrix
    # TODO RNN

    # TODO trasformers


if __name__ == "__main__":
    main()
