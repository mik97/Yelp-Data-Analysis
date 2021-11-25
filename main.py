from classes.dataset import Dataset
import preprocessing as prep_utils
import const


def main():
    review_dataset_tasks()


def review_dataset_tasks():
    '''
        Task 1, 2 and 3 that uses the review dataset.
    '''
    # TODO some statistics and visualization

    task1_pipeline()


def task1_pipeline():
    # COMPLETE 1. create tokens training set -> save it csv file (column: index, preprocessed text, sentiment)
    # COMPLETE 2. create tokens validation set -> save it csv file (column: index, preprocessed text, sentiment)
    # COMPLETE 3. create tokens test set -> save it csv file (column: index, preprocessed text, sentiment)

    # Get review dataset balanced with sentiment
    review_data = Dataset('review', 'sentiment')
    review_data.split(['text'], 'sentiment', n_samples=500_000)

    # COMPLETE  train set: fit tokenizer and embedding matrix

    # TODO train set ->  get tokens using training tokenizer
    # TODO val set -> get tokens using training tokenizer
    # TODO test set -> get tokens using training tokenizer

    cleaned_train, max_train_len = prep_utils.preprocess_text(
        review_data.train_data[0]['text'])

    # get tokenizer (it's the same trained on train set also for val and test set)
    tokenizer = prep_utils.get_tokenizer(cleaned_train)

    # get embedded matrix based containing vectors from a pretrained dict, vectors are related only to words found in train sentences

    # glove.twitter.27B.100Dim
    e_matrix = prep_utils.get_embedding_matrix(const.word_embedding_file, 'task1',
                                               tokenizer, len(tokenizer.index_word)+1)

    for i in range(len(e_matrix)):
        print(e_matrix[i])


if __name__ == "__main__":
    main()
