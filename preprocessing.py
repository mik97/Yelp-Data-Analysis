import re
import pickle
import time
from typing import Counter

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

utility = {
    "wml": WordNetLemmatizer(),
    "stop_words_dict": Counter(stopwords.words('english'))
}

tokenize_counter = 0


def get_tokenizer(sentences: list[str]) -> Tokenizer:
    ''' 
        Sentences: preprocessed sentences 
    '''
    # cleaned_sentences, max_lenght = preprocess_text(sentences)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    return tokenizer


def preprocess_text(sentences: list[str]) -> list[str]:
    '''Do preprocessing on input sentences.

    Return (list of cleaned sentences, max sentence lenght)
    '''
    global n_sentences
    n_sentences = len(sentences)

    print('Processing phase: cleaning the sentences...')
    mod_sentences = sentences.map(lambda txt: txt.lower())

    print('\tDecontracted the contracted forms...')
    mod_sentences = mod_sentences.apply(decontract)

    print('\tGetting cleaned tokens...')

    cleaned_tokens = list(map(clean_sentence, mod_sentences))

    with open('first_cleaned_tokens_training.pkl', 'wb') as f:
        pickle.dump(cleaned_tokens, f)

    # max sentence len (useful for padding)

    print('\tDetokenizing the sentences...')
    word_detokenizer = TreebankWordDetokenizer()

    detokenized_texts = [
        word_detokenizer.detokenize(sentence) for sentence in cleaned_tokens]

    return detokenized_texts


def clean_sentence(sentence):
    ''' remove stop-words and non alpha, also lemmatize words'''
    global tokenize_counter
    tokenize_counter += 1

    if (tokenize_counter % 10_000) == 0:
        print(
            f"\t\t{tokenize_counter} sentences processed ({round(tokenize_counter*100/n_sentences, 2)}%)")

    return [utility['wml'].lemmatize(token) for token in word_tokenize(sentence) if token.isalpha() and token not in utility["stop_words_dict"]]
    # list(map(process_token, word_tokenize(sentence)))


# def process_token(token):
#     if utility["stop_words_dict"].get(token):
#         return utility['wml'].lemmatize(token)


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


# extract word embedding  -----------------

def extract_word_embedding(path):
    embedding_indexes = dict()

    print(f'Reading pretrained word embedding from {path}...')
    with open(path, encoding='utf8') as f:
        for index, line in enumerate(f):
            if (index % 1000 == 0):
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
    pickled_matrix_path = utils.pickled_embedding_matrix_file_name(task_name)

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
