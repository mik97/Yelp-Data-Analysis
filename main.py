# execution times
# File loaded and pickled in 7.5567114869753516 seconds
# File unpickled in 1.1141046166419983 minutes

import dataset_handler


def main():
    review_df = dataset_handler.initDataset('review')
    dataset_handler.analyze(review_df, 'review')

if __name__ == "__main__":
    main()