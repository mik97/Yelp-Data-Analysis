from classes.dataset import Dataset


def main():
    review_dataset_tasks()


def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''
    # TODO some statistics and visualization

    task1_pipeline()


def task1_pipeline():
    cleaned_sentences = None

    # tokens_path = utils.get_tokens_file('task1')
    # w2v_path = utils.get_w2v_file('task1')

    # TODO ? w2v training only on training set

    # TODO 1. create tokens training set -> save it csv file (column: index, preprocessed text, sentiment)
    # TODO 2. create tokens validation set -> save it csv file (column: index, preprocessed text, sentiment)
    # TODO 3. create tokens test set -> save it csv file (column: index, preprocessed text, sentiment)

    review_data = Dataset('review', 'sentiment')
    review_data.split(['text'], 'sentiment', n_samples=500_000)


if __name__ == "__main__":
    main()
