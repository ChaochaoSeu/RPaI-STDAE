import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))
import torch
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.scaler import ZScoreScaler
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils import get_regular_settings

from .arch import STDAE
from .runner import PreTrainRunner

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
InterChageName = 'QiLin'
sampling_granularity = 10
DATA_NAME = f'{InterChageName}_{sampling_granularity}min'  # Dataset name
timestep_oneday = int(24 * 60/sampling_granularity)

regular_settings = get_regular_settings(DATA_NAME)
# INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
INPUT_LEN = timestep_oneday * 1
OUTPUT_LEN = timestep_oneday * 1

TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = False
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

SELECTED_CHANNELS = [1, 2, 3, 4]
TARGET_FEATURES = [0]

# Model architecture and parameters
MODEL_ARCH = STDAE


MODEL_PARAM = {
    "patch_size":12,
    "in_channel":4,
    "embed_dim":96,
    "num_heads":4,
    "mlp_ratio":4,
    "dropout":0.1,
    "encoder_depth":4,
    "decoder_depth":1,
    "spatial":True,
    "mode":"pre-train"
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = PreTrainRunner
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
CFG.MODEL.NAME = MODEL_ARCH.__name__ + '_SAE'
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = SELECTED_CHANNELS
CFG.MODEL.TARGET_FEATURES = TARGET_FEATURES

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
    'baselines',
    'STDAELSTM',
    'pretrain_save',
    CFG.MODEL.NAME,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN),str("history_turn_reconstruct")])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr":0.0005,
    "weight_decay":0,
    "eps":1.0e-8,
    "betas":(0.9, 0.95)
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones":[25, 50, 75],
    "gamma":0.5
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 8
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 8
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 8
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
