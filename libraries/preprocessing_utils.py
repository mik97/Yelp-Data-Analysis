from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from libraries import utils
from typing import Counter

import nltk
import numpy as np

import libraries.filenames_generator as filenames

import re
import os

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

utilities = {
    "wml": WordNetLemmatizer(),
    "stop_words_dict": Counter(stopwords.words('english'))
}

tokenizer_counter = 0
n_sentences = 0


def preprocess_texts(sentences, path=None):
    '''Given a list of texts, return list of cleaned texts.

    Preprocessing tasks: lowercasing, decontractions, stop-word removing, lemmatization 

    Returns
        list of cleaned sentences
    '''
    if os.path.exists(path):
        print(f'Loading pickled cleaned sentences data from {path}...')
        return utils.load_pickled(path)

    global tokenizer_counter
    global n_sentences
    n_sentences = len(sentences)

    print('Processing phase: text preprocessing...')
    mod_sentences = sentences.map(lambda text: text.lower())

    print('\tDecontracting the contracted forms...')
    mod_sentences = mod_sentences.apply(decontract)

    print('\tTokenizing sentences, removing stop-words and lemmatizing...')
    cleaned_tokens = [remove_stopw_lemmatize(tokenize_sentence(
        mod_sentence)) for mod_sentence in mod_sentences]

    print('\tDetokenizing sentences...')
    word_detokenizer = TreebankWordDetokenizer()

    cleaned_texts = [
        word_detokenizer.detokenize(sentence) for sentence in cleaned_tokens]

    if path:
        utils.save_pickled(path, cleaned_texts)
        print(f'\tTokens saved at {path}')

    tokenizer_counter = 0
    n_sentences = 0

    return cleaned_texts


def tokenize_sentence(sentence):
    global tokenizer_counter
    tokenizer_counter += 1

    if (tokenizer_counter % 10_000) == 0:
        print(
            f"\t\t{tokenizer_counter} sentences tokenized ({round(tokenizer_counter*100/n_sentences, 2)}%)")

    return word_tokenize(sentence)


def remove_stopw_lemmatize(tokens):
    '''
    input: a sentence tokens

    remove stop-words and non alpha, also lemmatize words
    '''
    return [utilities['wml'].lemmatize(token) for token in tokens if token.isalpha() and token not in utilities["stop_words_dict"]]


def decontract(sentence):
    # specific
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "cannot", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)

    return sentence

#  -----------------utils for neural net modeling -------------------------


def get_tokenizer(sentences) -> Tokenizer:
    ''' 
        Sentences: preprocessed sentences 
    '''
    # cleaned_sentences, max_lenght = preprocess_text(sentences)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    return tokenizer


def _tokenize(sequences, tokenizer, max_len=None, path=None):
    ''' Return padded sequences of tokens'''
    tokens = tokenizer.texts_to_sequences(sequences)
    padded_tokens = pad_sequences(tokens, maxlen=max_len)

    if path != None:
        utils.save_pickled(path, padded_tokens)

    return padded_tokens


def get_padded_tokens(texts, set='', task=''):
    # path file for cleaned texts
    cleaned_texts_filepath = filenames.picked_cleaned_sentences(
        set, task)

    # # path file for tokens
    # tokens_filepath = utils.tokens_file_name(set, task)

    # if it doesnt'exist, we have to calculate it (we need the cleaned sentences)
    cleaned_texts = utils.load_pickled(cleaned_texts_filepath) if os.path.exists(cleaned_texts_filepath) else preprocess_texts(
        texts, path=cleaned_texts_filepath)

    tokenizer = get_tokenizer(cleaned_texts)

    return tokenizer


def get_set_tokens(texts, tokenizer, set='', task=''):
    # path file for cleaned texts
    cleaned_texts_file = filenames.picked_cleaned_sentences(
        set, task)

    # path file for tokens
    tokens_file = filenames.pickled_tokens(set, task)

    if os.path.exists(tokens_file):  # if they already exists, load them
        return utils.load_pickled(tokens_file)  # tokens

    # if it doesnt'exist, we have to calculate it (we need the cleaned sentences)
    cleaned_texts = utils.load_pickled(cleaned_texts_file) if os.path.exists(cleaned_texts_file) else preprocess_texts(
        texts, path=cleaned_texts_file)

    return _tokenize(cleaned_texts, tokenizer, path=tokens_file)


# --------- word embedding --------------

def extract_word_embedding(path):
    embedding_indexes = dict()

    print(f'Reading pretrained word embedding from {path}...')
    with open(path, encoding='utf8') as f:
        for index, line in enumerate(f):
            if (index % 50_000 == 0):
                print(f'\t{index} words loaded')

            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_indexes[word] = vector

    return embedding_indexes


def get_embedding_matrix(word_emb_path, task_name, tokenizer, vocab_size):
    ''' Params:
        w_embedding_path: path of the trained word embedding 
        tokenizer:
        vocab_size:
        task_name: use it for looking for the correspondig pickled file, if it exists
        '''
    ''' Create embedding matrix'''
    # task1_embedding_matrix.npy
    pickled_matrix_path = filenames.embedding_matrix_pkl_file_name(task_name)

    if not os.path.exists(pickled_matrix_path):
        embedding_indexes = extract_word_embedding(word_emb_path)
        embedding_matrix = np.zeros((vocab_size, 100))

        print("Creating embedding matrix...")
        not_found_words = 0

        for word, index in tokenizer.word_index.items():
            if index > vocab_size - 1:
                break
            else:
                embedding_vector = embedding_indexes.get(word)
                # words not found into the embedding it's represented by a vector of zeros
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
                else:
                    not_found_words += 1

        # tot: 100 = not_found : x
        print(f'\t{round((not_found_words * 100 / vocab_size), 2)}% words vector not found ({not_found_words} over {vocab_size})')

        # save the pickled version of the matrix
        np.save(pickled_matrix_path, embedding_matrix)

        print(
            f'...embedding matrix created (matrix pickled at {pickled_matrix_path})')
        return embedding_matrix
    else:
        print(
            f'Loading pickled embedding matrix from {pickled_matrix_path}...')
        embedding_matrix = np.load(pickled_matrix_path)
        print(f'...embedding matrix loaded')

        return embedding_matrix
