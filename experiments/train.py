# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
# import torch._dynamo

# torch.set_float32_matmul_precision('high')
# torch._dynamo.config.accumulated_cache_size_limit = 256
# torch._dynamo.config.cache_size_limit = 256  # or higher
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.optimize_ddp = False

import basicts

torch.set_num_threads(2)  # avoid high cpu avg usage


def parse_args():
    """
    Run data for 10min, 5min, 3min intervals, please modify in the corresponding parameter files
    """
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    # parser.add_argument('-c', '--cfg', default='examples/regular_config.py', help='training config')

    # ====== TCN ======
    # parser.add_argument('-c', '--cfg', default='baselines/TCN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/TCN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/TCN/XueBu.py', help='training config')

    # ====== GRU ======
    # parser.add_argument('-c', '--cfg', default='baselines/GRU/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/GRU/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/GRU/XueBu.py', help='training config')

    # ====== TGCN ======
    # parser.add_argument('-c', '--cfg', default='baselines/TGCN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/TGCN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/TGCN/XueBu.py', help='training config')

    # ====== DCRNN ======
    # parser.add_argument('-c', '--cfg', default='baselines/DCRNN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/DCRNN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/DCRNN/XueBu.py', help='training config')

    # ====== STGCN ======
    # parser.add_argument('-c', '--cfg', default='baselines/STGCN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STGCN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STGCN/XueBu.py', help='training config')

    # ====== GWNet ======
    # parser.add_argument('-c', '--cfg', default='baselines/GWNet/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/GWNet/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/GWNet/XueBu.py', help='training config')

    # ====== GWNet RampToRamp ======
    # parser.add_argument('-c', '--cfg', default='baselines/GWNet_RampToRamp/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/GWNet_RampToRamp/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/GWNet_RampToRamp/XueBu.py', help='training config')

    # ====== MTGNN ======
    # parser.add_argument('-c', '--cfg', default='baselines/MTGNN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MTGNN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MTGNN/XueBu.py', help='training config')

    # ====== iTransformer ======
    # parser.add_argument('-c', '--cfg', default='baselines/iTransformer/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/iTransformer/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/iTransformer/XueBu.py', help='training config')

    # ====== STNorm ======
    # parser.add_argument('-c', '--cfg', default='baselines/STNorm/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STNorm/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STNorm/XueBu.py', help='training config')

    # ====== AGCRN ======
    # parser.add_argument('-c', '--cfg', default='baselines/AGCRN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/AGCRN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/AGCRN/XueBu.py', help='training config')

    # ====== STPGNN ======
    # parser.add_argument('-c', '--cfg', default='baselines/STPGNN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STPGNN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STPGNN/XueBu.py', help='training config')

    # ====== STAEformer ======
    # parser.add_argument('-c', '--cfg', default='baselines/STAEformer/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STAEformer/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STAEformer/XueBu.py', help='training config')

    # ====== D2STGNN ======
    # parser.add_argument('-c', '--cfg', default='baselines/D2STGNN/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/D2STGNN/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/D2STGNN/XueBu.py', help='training config')

    # ====== HimNet ======
    # parser.add_argument('-c', '--cfg', default='baselines/HimNet/QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/HimNet/DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/HimNet/XueBu.py', help='training config')

    # ====== STDMAE ======
    # STDMAE - QiLin: Uses turning traffic historical series + t_of_d, verified
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/SMAE_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/TMAE_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/STDMAE_QiLin.py', help='training config')

    # STDMAE - DanYangXinQu: Uses turning traffic historical series + t_of_d, verified
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/SMAE_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/TMAE_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/STDMAE_DanYangXinQu.py', help='training config')

    # STDMAE - XueBu: Uses turning traffic historical series + t_of_d, verified
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/SMAE_XueBu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/TMAE_XueBu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDMAE/STDMAE_XueBu.py', help='training config')

    # ====== STDAE ======
    # STDAE - QiLin
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/SAE_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/TAE_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/STDAEGWnet_QiLin.py', help='training config')

    # STDAE - DanYangXinQu
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/SAE_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/TAE_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/STDAEGWnet_DanYangXinQu.py', help='training config')

    # STDAE - XueBu
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/SAE_XueBu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/TAE_XueBu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAE/STDAEGWnet_XueBu.py', help='training config')

    # ====== STDAE Ablation ======
    # STDAEGWNET - QiLin
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAEGWnet_QiLin.py', help='training config')

    # STDAEGWNET - DanYangXinQu
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAEGWnet_DanYangXinQu.py', help='training config')

    # STDAEGWNET - XueBu
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAEGWnet_XueBu.py', help='training config')

    # STDAELSTM
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAELSTM_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAELSTM_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAELSTM_XueBu.py', help='training config')

    # STDAED2STGNN
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAED2STGNN_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAED2STGNN_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAED2STGNN_XueBu.py', help='training config')

    # STDAESTGCN
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAESTGCN_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAESTGCN_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAESTGCN_XueBu.py', help='training config')

    # STDAESTAEformer
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAESTAEformer_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAESTAEformer_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/STDAEAblation/STDAESTAEformer_XueBu.py', help='training config')

    # ====== MaskSTDAE ======
    # MaskSTDAE - QiLin: Mask strategy "East" + [-6:]
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskSAE_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskTAE_QiLin.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskSTDAEGWnet_QiLin.py', help='training config')

    # MaskSTDAE - DanYangXinQu: Mask strategy "East" + [-6:]
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskSAE_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskTAE_DanYangXinQu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskSTDAEGWnet_DanYangXinQu.py', help='training config')

    # MaskSTDAE - XueBu: Mask strategy "East" + [-6:]
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskSAE_XueBu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskTAE_XueBu.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='baselines/MaskSTDAE/MaskSTDAEGWnet_XueBu.py', help='training config')

    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()


def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()