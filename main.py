# execution times
# File loaded and pickled in 7.5567114869753516 seconds
# File unpickled in 1.1141046166419983 minutes

import os
import time
import const

import pandas as pd

def main():
    df = importDataset()

def importDataset():
    start_time = time.time()

    total = []
    final_df = []
    
    if os.path.exists('./data/review_data.pkl'):
        print("Unpickling existing file . . .")

        final_df = pd.read_pickle('./data/review_data.pkl')
        
        print("File unpickled in {0} minutes".format((time.time() - start_time)/60))
    else:
        print("Loading file . . .")
        # remember: each chunk is a regular dataframe object
        # 864 chunks w chunk == 10_000

        i = 1
        for chunk in pd.read_json(const.filesPath["review"], lines=True, orient="records", chunksize=10_000):
            if (i % 50 == 0):
                print("\t{0} chunks loaded")
            total.append(chunk)

            i = i + 1

        final_df = pd.concat(total, ignore_index= True)

        print("File loaded {0} minutes".format((time.time()-start_time )/60))

        pd.to_pickle(final_df, "./data/review_data.pkl")
        
    print(final_df.shape)
    print(final_df.head(100))

    return final_df

if __name__ == "__main__":
    main()


# preprocessing