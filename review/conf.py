import torch
import torch.nn.functional as F
import torchmetrics

import sys
sys.path.insert(0, r'E:\Clean\Lec\Btcmp\Practice\gitPrac')
from review.nn import ANN


config =\
    {
    "input": {
        "train": "../review/open/train.csv",
        "test": "../review/open/test.csv",
    },
     "output": {
        "prep_train": "../review/data/prep_train.csv",
        "test": "../review/open/test.csv",
    },
    "columns": {
        "use_trn_cols": ["ID", "사고일시", "요일", "기상상태", "시군구", "도로형태", "노면상태", "사고유형"],
        "drop_trn_flag": True,
        "drop_trn_cols": [
            "사고유형 - 세부분류",
            "법규위반",
            "가해운전자 차종",
            "가해운전자 성별",
            "가해운전자 연령",
            "가해운전자 상해정도",
            "피해운전자 차종",
            "피해운전자 성별",
            "피해운전자 연령",
            "피해운전자 상해정도",
            "사망자수",
            "중상자수",
            "경상자수",
            "부상자수",
        ],
        "use_tst_cols": ["ID", "사고일시", "요일", "기상상태", "시군구", "도로형태", "노면상태", "사고유형"],
        "drop_tst_flag": False,
        "drop_tst_cols": [],
        "index_col": "ID",
        "target_cols": "ECLO",
    },
    "model" : ANN,
    "model_param": {
        "input_dim" : 5,
        "hidden_dim" : [128, 128, 64, 32],
        "activation" : "sigmoid",
        "use_drop:bool" : True,
        "drop_ratio" : 0.0,
        "embed_cols_len" : 0, # 차원 늘릴 원본 컬럼의 갯수
        "embed_dim" : 10, # 늘릴 차원
        "output_dim" : 1,
    },
    "train_param":{
        'batch_size': 32,
        'shuffle': True,
        'epochs': 100,
        'metric': torchmetrics.MeanSquaredError(squared=False),
        'loss': F.mse_loss,
        
    },
    "optim_param": {
        "optim": torch.optim.Adam,
        "lr" : 0.001,
    },
    
    "device" : 'cpu'
}
