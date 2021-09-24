import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from gensim.utils import lemmatize
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')


def text_pipeline(sentences):
    # casing the characters
    mod_sentences = to_lower(sentences)
    # handling contracted forms
    mod_sentences = mod_sentences.apply(decontract)
    
    tokens = get_tokens(mod_sentences)
    # remove stopwords and non alpha num characters
    tokens = remove_stopwords_and_noalpha(tokens)

    # lemmatize
    lemm = to_lemmas(tokens)

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


def to_lemmas(data):
    toRet = []
    # for array in data:
    #     toRet.append(lemmatize(array))
    wml = WordNetLemmatizer()
    toRet.append(wml.lemmatize('went'))
    return toRet
