from math import pi
import os
from classes.dataset import Dataset
import preprocessing as prep_utils

import const
import utils
import pickle


def task1_pipeline():
    review_data = Dataset('review', 'sentiment')
    review_data.split(['text'], 'sentiment', n_samples=500_000)

    # preprocess text -> list[str]
    # fit tokenizer
    # tokenize(preprocessed text)
    # create embedding matrix

    # TODO fix training prep
    # ---------
    cleaned_train = prep_utils.preprocess_text(
        review_data.train_data[0]['text'])

    # get tokenizer (it's the same trained on train set also for val and test set)
    tokenizer = prep_utils.get_tokenizer(cleaned_train)

    train_tokens = prep_utils.tokenize(cleaned_train, tokenizer)

    # ---------
    test_tokens = get_set_tokens(
        review_data.test_data[0]['text'], tokenizer, set='test', task='task1')

    val_tokens = get_set_tokens(
        review_data.val_data[0]['text'], tokenizer, set='val', task='task1')

    # ---------
    # get embedded matrix based containing vectors from a pretrained dict, vectors are related only to words found in train sentences
    # glove.twitter.27B.100Dim
    e_matrix = prep_utils.get_embedding_matrix(const.word_embedding_file, 'task1',
                                               tokenizer, len(tokenizer.index_word)+1)

    # TODO train set ->  get tokens using training tokenizer
    # TODO val set -> get tokens using training tokenizer
    # TODO test set -> get tokens using training tokenizer

    # TODO RNN

    # TODO trasformers


def get_set_tokens(texts, tokenizer, set='', task=''):
    # path file for cleaned texts
    cleaned_texts_file = utils.get_cleaned_sen_file(
        set, task)

    # path file for tokens
    tokens_file = utils.get_tokens_file(set, task)

    if os.path.exists(tokens_file):  # if they already exists, load them
        return pickle.load(tokens_file)  # tokens

    # if it doesnt'exist, we have to calculate it (we need the cleaned sentences)
    cleaned_texts = pickle.load(cleaned_texts_file) if os.path.exists(cleaned_texts_file) else prep_utils.preprocess_text(
        texts, path=cleaned_texts_file)

    return prep_utils.tokenize(cleaned_texts, tokenizer)
