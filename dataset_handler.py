import os
import time
import const

import pandas as pd

def initDataset(type):
    '''
        Load the entire dataset and pickle it or read it directly from an existing file containing a pickled dataframe.

        Parameters:
            type (string): "business", "checkin", "review", "tip" or "user".
    '''
    start_time = time.time()
    
    df_toRet = []
    
    pkl_filepath = const.pklPath + type +'_data.pkl'
    
    if os.path.exists(pkl_filepath):
        print("Unpickling existing file . . .")
        df_toRet = pd.read_pickle(pkl_filepath)    
        print("File unpickled in {0} minutes".format((time.time() - start_time)/60))
    else:
        print("Loading file . . .")
        total = []

        # remember: each chunk is a regular dataframe object
        # 864 chunks w chunk == 10_000
        index = 1
        for chunk in pd.read_json(const.filesPath[type], lines=True, orient="records", chunksize=10_000):
            if (index % 50 == 0):
                print("\t{0} chunks loaded".format(index))

            total.append(chunk)

            index = index + 1

        df_toRet = pd.concat(total, ignore_index= True)

        print("File loaded {0} minutes".format((time.time() - start_time)/60))

        pd.to_pickle(df_toRet, pkl_filepath)
    
    print("Loaded file with {0} rows and {1} columns".format(df_toRet.shape[0], df_toRet.shape[1]))
    return df_toRet
