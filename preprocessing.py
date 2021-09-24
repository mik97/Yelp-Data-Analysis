import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from gensim.utils import lemmatize
from nltk.stem import WordNetLemmatizer


def to_lower(data):
    return data['text'].map(lambda txt: txt.lower())


def decontract(data):
    # specific
    data = re.sub(r"won\'t", "will not", data)
    data = re.sub(r"can\'t", "cannot", data)

    # general
    data = re.sub(r"n\'t", " not", data)
    data = re.sub(r"\'re", " are", data)
    data = re.sub(r"\'d", " would", data)
    data = re.sub(r"\'ll", " will", data)
    data = re.sub(r"\'ve", " have", data)
    data = re.sub(r"\'m", " am", data)

    return data


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
