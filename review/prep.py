import pandas as pd
from conf import config


df_train = pd.read_csv(config['input']['train'])
df_test = pd.read_csv(config['input']['test'])

if config['columns']['drop_trn_flag'] == True:
    df_train.drop(columns=config['columns']['drop_trn_cols'], axis=1, inplace=True)

if config['columns']['drop_tst_flag'] == True:
    df_test.drop(columns=config['columns']['drop_tst_cols'], axis=1, inplace=True)
    
