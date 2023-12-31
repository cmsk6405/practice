import torch
import pandas as pd
import numpy as np
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import trange
from torch.utils.data.dataset import TensorDataset 

import sys
sys.path.insert(0, r'E:\Clean\Lec\Btcmp\Practice\gitPrac')
from review.nn import ANN

from conf import config


def train_one_epoch(
    model:nn.Module,
    criterion:callable,
    optimizer:torch.optim.Optimizer,
    data_loader:DataLoader,
    metric:torchmetrics.Metric,
    device:str
    ) -> None:
    '''train one epoch
    
    Args:
        model: model
        criterion: loss
        optimizer: optimizer
        data_loader: data loader
        device: device
    '''
    model.train()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric.update(output, y)
            
class Train():
    '''
    작성예정
    '''
    def __init__(self):
        self.df_x = torch.tensor(pd.read_csv(config['output']['prep_train'], index_col=0).to_numpy(dtype=np.float32))
        
        self.df_y = torch.tensor(pd.read_csv("../review/data/prep_target.csv").to_numpy(dtype=np.float32))
            
    def train(self):
        device = torch.device(config['device'])
        ds = TensorDataset(self.df_x, self.df_y)
        dl = DataLoader(ds, batch_size = 32, shuffle = True)
        
        Model = ANN #config['model']
        # 일단 돌아가게 하고 나중에 설정
        # model_params = config['']['']   #cfg.get('model_params')
        # model_params['input_dim'] = self.df_x.shape[-1]
        model = Model(self.df_x.shape[-1]).to(device)
        
        Optim = config['optim_param']['optim']
        optim_params = config['optim_param']['lr']
        optimizer = Optim(model.parameters(),lr = optim_params)

        loss = config['train_param']['loss']
        metric = config['train_param']['metric']
        values = []
        pbar = trange(config['train_param']['epochs'])
        for _ in pbar:
            train_one_epoch(model, loss, optimizer, dl, metric, device)
            values.append(metric.compute().item())
            metric.reset()
            pbar.set_postfix(trn_loss=values[-1])
        # torch.save(model.state_dict(), files.get('output'))

a=Train()
a.train()