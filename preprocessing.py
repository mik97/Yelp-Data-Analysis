import os
import re
import pickle


from time import time
from gensim.models import Word2Vec

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# nltk.download('stopwords')
# nltk.download('wordnet')


def preprocess_text(sentences, name=''):
    print('Cleaning the sentences...')
    # casing the characters
    mod_sentences = to_lower(sentences)
    print('Decontracted the contracted forms...')
    # handling contracted forms
    mod_sentences = mod_sentences.apply(decontract)

    print('Getting tokens...')
    tokens = get_tokens(mod_sentences)
    print('Removing stopwords and non alpha words...')
    # remove stopwords and non alpha num characters
    tokens = remove_stopwords_and_noalpha(tokens)
    # lemmatize
    print('Lemmatizing...')
    lemm = to_lemmas(tokens)

    with open('./data/tokens_{0}.pkl'.format(name), 'wb') as file:
        pickle.dump(lemm, file)

    return lemm


def load_preprocessed_text(name=''):
    lemm = None

    with open('./data/tokens_{0}.pkl'.format(name), 'rb') as file:
        start_time = time()
        print('Open cleaned tokens file . . .')
        lemm = pickle.load(file)
        print('. . . tokens loaded in {0}'.format((time() - start_time)/60))

    return lemm


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


def remove_stopwords_and_noalpha(data):
    stop_words = stopwords.words('english')

    toRet = []
    for array in data:
        toRet.append(
            [word for word in array if word not in stop_words and word.isalpha()])

    return toRet

# def remove_stopwords_and_noalpha_and_lemmatize(data):


def process_tokens(data):
    stop_words = stopwords.words('english')
    wml = WordNetLemmatizer()

    toRet = []
    for array in data:
        toRet.append(
            [wml.lemmatize(word) for word in array if word not in stop_words and word.isalpha()])

    return toRet


def to_lemmas(data):
    toRet = []

    wml = WordNetLemmatizer()

    for array in data:
        toRet.append([wml.lemmatize(word) for word in array])

    return toRet


def get_word_embedding(sentences, name=''):
    model_path = './w2v_models/w2v_{0}.bin'.format(name)

    model = None

    if not os.path.exists(model_path):
        start_time = time()
        print('Creating word2vec model . . .')

        model = Word2Vec(sentences, min_count=1)

        model.save(model_path)

        print('. . .model saved succesfully in {0} minutes'.format(
            (time() - start_time)/60))
    else:
        print('Loading existing word2vec model. . .')

        start_time = time()
        model = Word2Vec.load(model_path)

        print('. . . model loaded successfully in {0} minutes'.format(
            (time() - start_time)/60))

    return model
