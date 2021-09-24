# execution times
# File loaded and pickled in 7.5567114869753516 seconds
# File unpickled in 1.1141046166419983 minutes

from os import path
from time import time
from nltk import text
import pandas as pd
import dataset_handler
import preprocessing


def main():
    # review_df = dataset_handler.initDataset('review')
    # dataset_handler.analyze(review_df, 'review')
    # new_df = dataset_handler.get_balanced_subset(review_df, 'sentiment')
    # new_df.head(100).to_csv(path_or_buf='test.csv')
    df = pd.read_csv("./test.csv")
    print(df)
    start_time = time()

    texts = preprocessing.to_lower(df)
    print(texts)
    texts = texts.apply(preprocessing.decontract)
    print(texts)
    tokens = preprocessing.get_tokens(texts)
    print("%s seconds" % (time() - start_time))
    # print(tokens[1])
    # lowerTokens = list(map(preprocessing.toLowerCase, tokens))
    # print(lowerTokens[1])
    start_time = time()
    stop_words = preprocessing.remove_stopwords_and_noalpha(tokens)
    print("%s seconds" % (time() - start_time))
    print(stop_words)
    lemm = preprocessing.to_lemmas(stop_words)
    print(lemm)


if __name__ == "__main__":
    main()
