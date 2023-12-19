import pandas as pd
import numpy as np
from conf import config
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from datasets import HomeData
import copy

# @dataclass
class HomeData():
    '''
    작성예정
    '''
    
    def __init__(self):
        self.train_csv = pd.read_csv(config["input"]["train"])
        self.test_csv = pd.read_csv(config["input"]["test"])
                
    
    def Preprocess(self):
        df_train = copy.deepcopy(self.train_csv)
        df_test = copy.deepcopy(self.test_csv)

        if config["columns"]["drop_trn_flag"] is True:
            df_train.drop(columns=config["columns"]["drop_trn_cols"], axis=1, inplace=True)
            print("train drop ok")
        if config["columns"]["drop_tst_flag"] is True:
            df_test.drop(columns=config["columns"]["drop_tst_cols"], axis=1, inplace=True)
            
        df_train = HomeData().CustomPrep(df_train)
        df_test = HomeData().CustomPrep(df_test)
        print("here ok")
        
        df_num = df_train.select_dtypes(include=["object"])
        
        df_cat = df_train.select_dtypes(include=["object"])
        enc = OneHotEncoder(
            dtype=np.float32,
            sparse_output=False,
            drop="if_binary",
            handle_unknown="ignore",
        )
        enc.fit(df_cat)
        df_cat_onehot = pd.DataFrame(
            enc.transform(df_cat), columns=enc.get_feature_names_out()
        ).reset_index(drop=True)

        return pd.concat([df_num, df_cat_onehot], axis=1).set_index(df_train.index)
            
        
        

    #사용하는 컬럼
    #"기상상태" "도로형태", "노면상태",],
    #사고유형 -차대사람, 차대차, 차량단독
    
    
    def CustomPrep(self, df):
        if '사고일시' in df.columns:
            df['날짜'] = df['사고일시'].str.split(' ').str[0]
            df['연'] = df['날짜'].str.split('-').str[0]
            df['월'] = df['날짜'].str.split('-').str[1]
            df['일'] = df['날짜'].str.split('-').str[2]
            df.drop('날짜', axis=1, inplace=True)
            df['시간'] = df['사고일시'].str.split(' ').str[1]
            df.drop('사고일시', axis=1, inplace=True)
            
        if '요일' in df.columns:
            df['요일'] = df['요일'].map(lambda x:
            1 if x == '토요일' or '일요일' else
            0
            )
            
        if '사고일시' in df.columns:    
            df['군'] = df['시군구'].str.split(' ').str[1]
            df['동'] = df['시군구'].str.split(' ').str[2]
            df.drop('시군구', axis=1, inplace=True)
            
        return df

    def Encoding(self, df):
        label_encoder = LabelEncoder()
        encoded_data = label_encoder.fit_transform(df.columns)
        
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        onehot_encoded_data = onehot_encoder.fit_transform(encoded_data.reshape(-1, 1))
        df_encoded = pd.DataFrame(onehot_encoded_data, columns=[f'category_{i}' for i in range(onehot_encoded_data.shape[1])])
        
        return df
    


    
#실행용
a = HomeData()
a.Preprocess()