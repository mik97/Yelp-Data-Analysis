import re
import pickle
import time

import utils
import numpy as np


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nltk.download('stopwords')
# nltk.download('wordnet')


def get_tokenizer(sentences: list[str]):
    ''' sentences: preprocessed sentences 

        return fited tokenizer
    '''
    # cleaned_sentences, max_lenght = preprocess_text(sentences)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    return tokenizer


def preprocess_text(sentences: list[str]) -> tuple[list[str], int]:
    '''Do preprocessing on input sentences

    Return (list of cleaned sentences, max sentence lenght)'''

    print('Processing phase: cleaning the sentences...')
    mod_sentences = sentences.map(lambda txt: txt.lower())

    print('\tDecontracted the contracted forms...')
    mod_sentences = mod_sentences.apply(decontract)

    print('\tGetting initial tokens...')
    first_tokens = mod_sentences.apply(word_tokenize)

    print('\tRemoving stopwords and non alpha words, lemmatizing remaining words...')
    lemmas = retrieve_lemmas(first_tokens)
    # max sentence len (useful for padding)
    max_len = len(max(lemmas, key=len))

    print('\Detokenizing the sentences...')
    word_detokenizer = TreebankWordDetokenizer()

    detokenized_texts = [
        word_detokenizer.detokenize(sentence) for sentence in lemmas]

    return detokenized_texts, max_len


def tokenize(sequences: list[str], tokenizer: Tokenizer, max_len: int = None) -> np.ndarray:
    ''' Return padded sequences of tokens'''
    tokens = tokenizer.texts_to_sequences(sequences)
    return pad_sequences(tokens, maxlen=max_len)

# def load_preprocessed_text(file_path):
#     with open(file_path, 'rb') as file:
#         start_time = time.time()

#         print(f'Load cleaned tokens file {file_path}...')
#         tokens = pickle.load(file)
#         print(f'...tokens loaded in {utils.get_minutes(start_time)}')

#     return tokens


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


def retrieve_lemmas(sentences_tokens: list[list[str]]) -> list[list[str]]:
    ''' remove stop-words and non alpha, also lemmatize words'''
    wml = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    # toRet = []
    # for array in data:
    #     toRet.append(
    #         [[wml.lemmatize(word) for word in array if word not in stop_words and word.isalpha()] for array in data ])

    # return toRet

    return [[wml.lemmatize(word) for word in array if word.isalpha() and word not in stop_words] for array in sentences_tokens]


# extract word embedding  -----------------

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
