import os
import re
import pickle
import time
import utils

from gensim.models import Word2Vec

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# nltk.download('stopwords')
# nltk.download('wordnet')


def preprocess_text(sentences, file_path):
    print('Processing phase: cleaning the sentences...')

    mod_sentences = to_lower(sentences)

    print('\tDecontracted the contracted forms...')
    mod_sentences = mod_sentences.apply(decontract)

    print('\tGetting tokens...')
    tokens = get_tokens(mod_sentences)

    print('\tRemoving stopwords and non alpha words, lemmatizing remaining wordss...')
    final_tokens = process_tokens(tokens)

    # pickled lemmas and
    with open(file_path, 'wb') as file:
        pickle.dump(final_tokens, file)

    return final_tokens


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


# def remove_stopwords_and_noalpha(data):
#     stop_words = stopwords.words('english')

#     toRet = []
#     for array in data:
#         toRet.append(
#             [word for word in array if word not in stop_words and word.isalpha()])

#     return toRet


def process_tokens(data):
    ''' remove stop-words and non alpha, also lemmatize words'''
    stop_words = stopwords.words('english')
    wml = WordNetLemmatizer()

    toRet = []
    for array in data:
        toRet.append(
            [wml.lemmatize(word) for word in array if word not in stop_words and word.isalpha()])

    return toRet


# def to_lemmas(data):
#     toRet = []

#     wml = WordNetLemmatizer()

#     for array in data:
#         toRet.append([wml.lemmatize(word) for word in array])

#     return toRet


def get_word_embedding(sentences, model_path):
    model = None

    start_time = time.time()

    if not os.path.exists(model_path):

        print(f'Creating word2vec model at {model_path}...')
        model = Word2Vec(sentences, min_count=1)
        model.save(model_path)

        print(
            f'...model saved succesfully in {utils.get_minutes(start_time)} minutes')
    else:
        print(f'Loading existing word2vec model at {model_path}...')

        model = Word2Vec.load(model_path)

        print(
            f'...model loaded successfully in {utils.get_minutes(start_time)} minutes')

    return model
