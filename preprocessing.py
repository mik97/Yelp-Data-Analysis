import os
import re
import pickle
import time

from nltk.util import pad_sequence
import utils
import numpy as np

# from gensim.models import Word2Vec

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from tensorflow.keras.preprocessing.text import Tokenizer

# TODO use keras tokenizer

# nltk.download('stopwords')
# nltk.download('wordnet')


def preprocess_text(sentences):
    print('Processing phase: cleaning the sentences...')
    mod_sentences = to_lower(sentences)

    print('\tDecontracted the contracted forms...')
    mod_sentences = mod_sentences.apply(decontract)

    print('\tGetting initial tokens...')
    first_tokens = get_tokens(mod_sentences)

    print('\tRemoving stopwords and non alpha words, lemmatizing remaining words...')
    lemmas = retrieve_lemmas(first_tokens)

    max_len = max(lemmas, key=len)

    wordDetok = TreebankWordDetokenizer()
    detokenized = [wordDetok.detokenize(words) for words in lemmas]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(detokenized)
    sequences = tokenizer.texts_to_sequences(detokenized)

    reviews = pad_sequence(sequences,max_len)

    return reviews, tokenizer


def load_preprocessed_text(file_path):
    with open(file_path, 'rb') as file:
        start_time = time.time()

        print(f'Load cleaned tokens file {file_path}...')
        tokens = pickle.load(file)
        print(f'...tokens loaded in {utils.get_minutes(start_time)}')

    return tokens


def to_lower(data):
    return data.map(lambda txt: txt.lower())


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


def get_tokens(data):
    return data.apply(word_tokenize)


def toLowerCase(array):
    return [token.lower() for token in array]


def retrieve_lemmas(data):
    ''' remove stop-words and non alpha, also lemmatize words'''
    stop_words = stopwords.words('english')
    wml = WordNetLemmatizer()

    toRet = []
    for array in data:
        toRet.append(
            [wml.lemmatize(word) for word in array if word not in stop_words and word.isalpha()])

    return toRet

# extract word embedding


def extract_word_embedding(path):
    # path = 'glove.6B/glove.6B.100d.txt'
    embedding_indexes = dict()
    with open(path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_indexes[word] = vector

    return embedding_indexes


def get_embedding_matrix(tokenizer, vocab_size, embedding_indexes):
    ''' Create embedding matrix'''
    embedding_matrix = np.zeros((vocab_size, 100))
    print("Creating embedding matrix...")
    for word, index in tokenizer.word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embedding_indexes.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    print(f'Computed embedding matrix: {embedding_matrix}')

    return embedding_matrix

    # # Maybe useless
    # def get_word_embedding(sentences, model_path):
    #     model = None

    #     start_time = time.time()

    #     if not os.path.exists(model_path):

    #         print(f'Creating word2vec model at {model_path}...')
    #         model = Word2Vec(sentences, min_count=1)
    #         model.save(model_path)

    #         print(
    #             f'...model saved succesfully in {utils.get_minutes(start_time)} minutes')
    #     else:
    #         print(f'Loading existing word2vec model at {model_path}...')

    #         model = Word2Vec.load(model_path)

    #         print(
    #             f'...model loaded successfully in {utils.get_minutes(start_time)} minutes')

    #     return model
