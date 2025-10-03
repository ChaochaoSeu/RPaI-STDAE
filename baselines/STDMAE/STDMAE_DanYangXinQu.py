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

from .arch import STDMAE

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
InterChageName = 'DanYangXinQu'
sampling_granularity = 3
DATA_NAME = f'{InterChageName}_{sampling_granularity}min'  # Dataset name
timestep_oneday = int(24 * 60/sampling_granularity)

regular_settings = get_regular_settings(DATA_NAME)
# INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
INPUT_LEN = timestep_oneday * 1

OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence

TRAIN_VAL_TEST_RATIO = [17/23, 1-17/23-(3*timestep_oneday-12+INPUT_LEN)/(23*timestep_oneday), (3*timestep_oneday-12+INPUT_LEN)/(23*timestep_oneday)]

NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

SELECTED_CHANNELS = [0,10]
TARGET_FEATURES = [0]

# Model architecture and parameters
MODEL_ARCH = STDMAE
adj_mx, _ = load_adj("datasets/" + DATA_NAME + "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "dataset_name": DATA_NAME,
    "pre_trained_tmae_path": "baselines/STDMAE/mask_save/Mask_TMAE/DanYangXinQu_3min_100_480_12/b18274680e117122391b7b8fc0327340/Mask_TMAE_best_val_MAE.pt",
    "pre_trained_smae_path": "baselines/STDMAE/mask_save/Mask_SMAE/DanYangXinQu_3min_100_480_12/155e6e7c75c761af85de574a0fe3c5fa/Mask_SMAE_best_val_MAE.pt",
    "mask_args": {
                    "patch_size":12,
                    "in_channel":1,
                    "embed_dim":96,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "mask_ratio":0.25,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "mode":"forecasting"
    },
    "backend_args": {
    "num_nodes": 12,
    "supports": [torch.tensor(i) for i in adj_mx],
    "dropout": 0.3,
    "gcn_bool": True,
    "addaptadj": True,
    "aptinit": None,
    "in_dim": 2,
    "out_dim": 12,
    "residual_channels": 32,
    "dilation_channels": 32,
    "skip_channels": 64,
    "end_channels": 128,
    "kernel_size": 2,
    "blocks": 4,
    "layers": 2
    },
    "short_term_len": OUTPUT_LEN
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
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
# 提前停止
CFG.TRAIN.EARLY_STOPPING_PATIENCE = 10 # 提前停止的耐心值。默认值：None。如果未指定，则不会使用提前停止。

CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr":0.001,
    "weight_decay":1.0e-4,
    "eps":1.0e-8,
}

# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones":[1, 5, 10, 25, 50, 75],
    "gamma":0.5
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 256
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 1
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
# curriculum learning
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 6
CFG.TRAIN.CL.PREDICTION_LENGTH = 12

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 128
CFG.VAL.DATA.NUM_WORKERS = 1
CFG.VAL.DATA.PIN_MEMORY = True

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 128
CFG.TEST.DATA.NUM_WORKERS = 1
CFG.TEST.DATA.PIN_MEMORY = True

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.HORIZONS = [3, 6, 12] # Prediction horizons for evaluation. Default: []
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
