from dataclasses import dataclass
from torch.utils.data import Dataset
import pandas as pd
# from typing import Literal
# from sklearn.preprocessing import OneHotEncoder

# from conf import config


@dataclass
class HomeData(Dataset):
    
    '''
    안되면 데이터클래스 지우고
    '''
    df_train: pd.DataFrame
    
    
    # def __init__(self, df_train, df_test):
    #     self.df_train = df_train
    #     self.df_test = df_test

    def __len__(self):
        return len(self.df_train)

    def __getitem__(self, index):
        # 예시로 df_train의 인덱스에 해당하는 샘플을 반환
        sample = self.df_train.iloc[index]
        return sample
    
  
    def preprocess(self):
        print('ok')
        print(df_train)
        # trn_df, target, tst_df = self._get_dataset()
        df_train = self.df_train
        trn_df = df_train
        target = df_train
        return trn_df, target
        
        
        
        
  # def one_hot_enconding():
    #     encoder = OneHotEncoder(sparse=False)
    
    # def _read_df(self, split:Literal['train', 'test']='train'):
    #     if split == 'train':
    #         df = pd.read_csv(self.file_trn, index_col=self.index_col)
    #         df.dropna(axis=0, subset=[self.target_col], inplace=True)
    #         target = df[self.target_col]
    #         df.drop([self.target_col], axis=1, inplace=True)
    #         return df, target
    #     elif split == 'test':
    #         df = pd.read_csv(self.file_tst, index_col=self.index_col)
    #         return df
    #     raise ValueError(f'"{split}" is not acceptable.')
    
    