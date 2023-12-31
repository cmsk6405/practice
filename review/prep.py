import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from datasets import HomeData

from conf import config

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
            
        df_train = self.CustomPrep(df_train)
        df_test = self.CustomPrep(df_test)
        
        #트레인데이터만 인코딩함
        prep_train = self.Encoding(df_train)

        return prep_train
            
        
        

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
            
        if '시군구' in df.columns:
            df['군'] = df['시군구'].str.split(' ').str[1]
            df['동'] = df['시군구'].str.split(' ').str[2]
            df.drop('시군구', axis=1, inplace=True)
            
        return df

    def Encoding(self, df):
        
        idx = config["columns"]["index_col"]
        tgt = config["columns"]["target_cols"]
        df_idx = df[idx]
        df_tgt = df[tgt]
        df.drop(idx,axis=1, inplace= True)
        df.drop(tgt,axis=1, inplace= True)
        print('encoding cols: ',df.columns)
        
        # df.drops("ID")
        # Numeric
        df_num = df.select_dtypes(include=["number"])
        # if self.fill_num_strategy == "mean":
        #     fill_values = df_num.mean(axis=1)
        # elif self.fill_num_strategy == "min":
        #     fill_values = df_num.min(axis=1)
        # elif self.fill_num_strategy == "max":
        #     fill_values = df_num.max(axis=1)
        # df_num.fillna(fill_values, inplace=True)
        # df_num.reset_index(drop=True, inplace=True)
        # if self.x_scaler is not None:
        #     df_num = pd.DataFrame(self._scale_X(df_num), columns=df_num.columns)

        # Categorical
        df_cat = df.select_dtypes(include=["object"])
        enc = OneHotEncoder(
            dtype=np.float32,
            sparse_output=False,
            drop="if_binary",
            handle_unknown="ignore",
        )
        enc.fit(df_cat)
        df_cat_onehot = pd.DataFrame(
            enc.transform(df_cat), columns=enc.get_feature_names_out()).reset_index(drop=True)
        return pd.concat([df_num, df_cat_onehot,df_tgt], axis=1).set_index(df_idx)
        


if __name__ == "__main__":
#실행용
    prep_class = HomeData()
    processed_data = prep_class.Preprocess()
    print(processed_data.columns)
    processed_data.to_csv(config["output"]["prep_train"])