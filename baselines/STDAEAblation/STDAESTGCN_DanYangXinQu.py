import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils import get_regular_settings, load_adj

from .arch import STDAESTGCN

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
InterChageName = 'DanYangXinQu'
sampling_granularity = 5

is_use_tae = True
is_use_sae = True

DATA_NAME = f'{InterChageName}_{sampling_granularity}min'  # Dataset name
timestep_oneday = int(24 * 60/sampling_granularity)

regular_settings = get_regular_settings(DATA_NAME)
# INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
INPUT_LEN = timestep_oneday * 1

OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence # 10min 采样粒度
TRAIN_VAL_TEST_RATIO = [17/23, 1-17/23-(3*timestep_oneday-12+INPUT_LEN)/(23*timestep_oneday), (3*timestep_oneday-12+INPUT_LEN)/(23*timestep_oneday)]

NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

SELECTED_CHANNELS = [1, 2, 3, 4]
TARGET_FEATURES = [0]

# Model architecture and parameters
MODEL_ARCH = STDAESTGCN
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "normlap")
adj_mx = torch.Tensor(adj_mx[0])

MODEL_PARAM = {
    "dataset_name": DATA_NAME,
    "pre_trained_tae_path": "baselines/STDAEAblation/pretrain_save/STDAE_TAE/DanYangXinQu_5min_100_288_288_history_turn_reconstruct/297c20e7f736bb4e39deeb7b231cfd67/STDAE_TAE_best_val_MAE.pt",
    "pre_trained_sae_path": "baselines/STDAEAblation/pretrain_save/STDAE_SAE/DanYangXinQu_5min_100_288_288_history_turn_reconstruct/9c2ca6eaac618603a82c9c8b7678dc0f/STDAE_SAE_best_val_MAE.pt",
    "stdae_args": {
                    "patch_size":12,
                    "in_channel":4,
                    "embed_dim":96,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "mode":"forecasting"
    },
    "stgcn_args": {
        "Ks": 3,
        "Kt": 3,
        "blocks": [[4], [64, 16, 64], [64, 16, 64], [128, 128], [OUTPUT_LEN]],
        "T": 12,
        "n_vertex": 12,
        "act_func": "glu",
        "graph_conv_type": "cheb_graph_conv",
        "gso": adj_mx,
        "bias": True,
        "droprate": 0.5
    },
    "short_term_len": OUTPUT_LEN,
    'is_use_tae': is_use_tae,
    'is_use_sae': is_use_sae
}

NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.RANDOM = torch.randint(0, 100000, (1,)).item()
############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'selected_channel': SELECTED_CHANNELS,
    'target_channel': TARGET_FEATURES,
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = SELECTED_CHANNELS
CFG.MODEL.TARGET_FEATURES = TARGET_FEATURES
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    CFG.MODEL.NAME,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN), f'TAE_{is_use_tae}', f'SAE_{is_use_sae}'])
)

CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0002,
    "weight_decay": 0.0003,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 25, 75],
    "gamma": 0.5
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 1
############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.NUM_WORKERS = 1
############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.NUM_WORKERS = 1
############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12] # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
