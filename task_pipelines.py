from math import pi
import os

from tensorflow.python.keras.preprocessing.text import Tokenizer
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

    train_tokens, tokenizer = get_set_tokens_tokenizer(
        review_data.train_data[0]['text'], set='train', task='task1')

    test_tokens = get_set_tokens(
        review_data.test_data[0]['text'], tokenizer, set='test', task='task1')

    val_tokens = get_set_tokens(
        review_data.val_data[0]['text'], tokenizer, set='val', task='task1')

    # ---------
    # get embedded matrix based containing vectors from a pretrained dict, vectors are related only to words found in train sentences
    # glove.twitter.27B.100Dim
    e_matrix = prep_utils.get_embedding_matrix(const.word_embedding_file, 'task1',
                                               tokenizer, len(tokenizer.index_word)+1)

    # TODO RNN

    # TODO trasformers


def get_set_tokens_tokenizer(texts, set='', task=''):
    # path file for cleaned texts
    cleaned_texts_file = utils.get_cleaned_sen_file(
        set, task)

    # path file for tokens
    tokens_file = utils.get_tokens_file(set, task)

    # if it doesnt'exist, we have to calculate it (we need the cleaned sentences)
    cleaned_texts = utils.load_pickled(cleaned_texts_file) if os.path.exists(cleaned_texts_file) else prep_utils.preprocess_text(
        texts, path=cleaned_texts_file)

    tokenizer = prep_utils.get_tokenizer(cleaned_texts)

    tokens = utils.load_pickled(tokens_file) if os.path.exists(
        tokens_file) else prep_utils.tokenize(cleaned_texts, tokenizer, path=tokens_file)

    return tokens, tokenizer


def get_set_tokens(texts, tokenizer, set='', task=''):
    # path file for cleaned texts
    cleaned_texts_file = utils.get_cleaned_sen_file(
        set, task)

    # path file for tokens
    tokens_file = utils.get_tokens_file(set, task)

    if os.path.exists(tokens_file):  # if they already exists, load them
        return utils.load_pickled(tokens_file)  # tokens

    # if it doesnt'exist, we have to calculate it (we need the cleaned sentences)
    cleaned_texts = utils.load_pickled(cleaned_texts_file) if os.path.exists(cleaned_texts_file) else prep_utils.preprocess_text(
        texts, path=cleaned_texts_file)

    return prep_utils.tokenize(cleaned_texts, tokenizer, path=tokens_file)
