from typing import Optional, Dict

import torch
from tqdm import tqdm
from easytorch.utils.dist import master_only

from basicts.runners import SimpleTimeSeriesForecastingRunner


class PreTrainRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        data = self.preprocessing(data)
        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]

        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)

        future_turn = self.select_target_features(history_data)

        # feed forward
        reconstruction = self.model(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch)
        results = {'prediction': reconstruction, 'target': future_turn, 'inputs': history_data}

        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:

        for data in tqdm(self.test_data_loader):
            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
            # metrics
            if not self.if_evaluate_on_gpu:
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()

            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(forward_return['prediction'], forward_return['target'], null_val=self.null_val)
                self.update_epoch_meter("test/"+metric_name, metric_item.item())


# import os
# import numpy as np
# from typing import Optional, Dict
#
# import torch
# from tqdm import tqdm
# from easytorch.utils.dist import master_only
#
# from basicts.runners import SimpleTimeSeriesForecastingRunner
#
#
# class PreTrainRunner(SimpleTimeSeriesForecastingRunner):
#     def __init__(self, cfg: dict):
#         super().__init__(cfg)
#
#     def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
#         """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.
#
#         Args:
#             data (tuple): data (future data, history data). [B, L, N, C] for each of them
#             epoch (int, optional): epoch number. Defaults to None.
#             iter_num (int, optional): iteration number. Defaults to None.
#             train (bool, optional): if in the training process. Defaults to True.
#
#         Returns:
#             tuple: (prediction, real_value)
#         """
#
#         data = self.preprocessing(data)
#         # Preprocess input data
#         future_data, history_data = data['target'], data['inputs']
#         history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
#         future_data = self.to_running_device(future_data)  # Shape: [B, L, N, C]
#
#         batch_size, length, num_nodes, _ = future_data.shape
#
#         # Select input features
#         history_data = self.select_input_features(history_data)
#
#         future_turn = self.select_target_features(history_data)
#
#         # feed forward
#         reconstruction = self.model(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch)
#         results = {'prediction': reconstruction, 'target': future_turn, 'inputs': history_data}
#
#         results = self.postprocessing(results)
#
#         return results
#
#     @torch.no_grad()
#     @master_only
#     def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
#
#         # 添加数据收集列表
#         all_predictions = []
#         all_targets = []
#         all_inputs = []
#
#         for data in tqdm(self.test_data_loader):
#             forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
#             # metrics
#             if not self.if_evaluate_on_gpu:
#                 forward_return['target'] = forward_return['target'].detach().cpu()
#                 forward_return['prediction'] = forward_return['prediction'].detach().cpu()
#                 forward_return['inputs'] = forward_return['inputs'].detach().cpu()
#
#             # 收集postprocessing后的数据
#             all_predictions.append(forward_return['prediction'])
#             all_targets.append(forward_return['target'])
#             all_inputs.append(forward_return['inputs'])
#
#             for metric_name, metric_func in self.metrics.items():
#                 metric_item = metric_func(forward_return['prediction'], forward_return['target'],
#                                           null_val=self.null_val)
#                 self.update_epoch_meter("test/" + metric_name, metric_item.item())
#
#         # 拼接所有batch的数据
#         all_predictions = torch.cat(all_predictions, dim=0)
#         all_targets = torch.cat(all_targets, dim=0)
#         all_inputs = torch.cat(all_inputs, dim=0)
#
#         # 保存数据到本地
#         self.save_test_results(all_predictions, all_targets, all_inputs, train_epoch)
#
#         # 返回收集的数据
#         return {
#             'prediction': all_predictions,
#             'target': all_targets,
#             'inputs': all_inputs
#         }
#
#     def save_test_results(self, predictions, targets, inputs, train_epoch=None):
#         """保存测试结果到本地"""
#
#         # 确保保存目录存在
#         save_dir = getattr(self, 'ckpt_save_dir', './test_results')
#         os.makedirs(save_dir, exist_ok=True)
#
#         # 转换为numpy数组
#         predictions_np = predictions.cpu().numpy()
#         targets_np = targets.cpu().numpy()
#         inputs_np = inputs.cpu().numpy()
#
#         # 生成文件名
#         if train_epoch is not None:
#             filename = f'test_results_epoch_{train_epoch}.npz'
#         else:
#             filename = 'test_results.npz'
#
#         # 保存为npz文件
#         save_path = os.path.join(save_dir, filename)
#         np.savez(save_path,
#                  predictions=predictions_np,
#                  targets=targets_np,
#                  inputs=inputs_np)
#
#         print(f"Test results saved to: {save_path}")
#         print(
#             f"Data shapes - Predictions: {predictions_np.shape}, Targets: {targets_np.shape}, Inputs: {inputs_np.shape}")