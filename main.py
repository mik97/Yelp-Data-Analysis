from classes.dataset import Dataset
from preprocessing import tokenize, preprocess_text


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

    # COMPLETE 1. create tokens training set -> save it csv file (column: index, preprocessed text, sentiment)
    # COMPLETE 2. create tokens validation set -> save it csv file (column: index, preprocessed text, sentiment)
    # COMPLETE 3. create tokens test set -> save it csv file (column: index, preprocessed text, sentiment)

    review_data = Dataset('review', 'sentiment')
    review_data.split(['text'], 'sentiment', n_samples=500_000)

    # TODO train set ->  fit tokenizer, get tokens, e embedding matrix
    # TODO val set -> get tokens using training tokenizer
    # TODO test set -> get tokens using training tokenizer

    preprocessed_text = preprocess_text(review_data.train_data[0])
    print("Preprocessed text: ", preprocessed_text)

    tokens = tokenize(preprocessed_text)
    print("Tokens: ", tokens)


if __name__ == "__main__":
    main()
