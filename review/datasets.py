from dataclasses import dataclass
import pandas as pd
from typing import Literal
from sklearn.preprocessing import OneHotEncoder

from conf import config


@dataclass
class HomeData(Dataset):
    def one_hot_enconding():
        encoder = OneHotEncoder(sparse=False)
    
    def _read_df(self, split:Literal['train', 'test']='train'):
        if split == 'train':
            df = pd.read_csv(self.file_trn, index_col=self.index_col)
            df.dropna(axis=0, subset=[self.target_col], inplace=True)
            target = df[self.target_col]
            df.drop([self.target_col], axis=1, inplace=True)
            return df, target
        elif split == 'test':
            df = pd.read_csv(self.file_tst, index_col=self.index_col)
            return df
        raise ValueError(f'"{split}" is not acceptable.')
    
    
    def preprocess(self):
        # trn_df, target, tst_df = self._get_dataset()
        trn_df, target = self._read_df('train')
        tst_df = self._read_df('test')
        
