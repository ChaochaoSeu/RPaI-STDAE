import json

import numpy as np
import torch

from .base_scaler import BaseScaler


class ZScoreScaler(BaseScaler):
    """
    ZScoreScaler performs Z-score normalization on the dataset, transforming the data to have a mean of zero 
    and a standard deviation of one. This is commonly used in preprocessing to normalize data, ensuring that 
    each feature contributes equally to the model.

    Attributes:
        mean (np.ndarray): The mean of the training data used for normalization. 
            If `norm_each_channel` is True, this is an array of means, one for each channel. Otherwise, it's a single scalar.
        std (np.ndarray): The standard deviation of the training data used for normalization.
            If `norm_each_channel` is True, this is an array of standard deviations, one for each channel. Otherwise, it's a single scalar.
        target_channel (int): The specific channel (feature) to which normalization is applied.
            By default, it is set to 0, indicating the first channel.
    """

    def __init__(self, dataset_name: str, train_ratio: float, selected_channel: list = None, target_channel: list = None , norm_each_channel: bool = True, rescale: bool = True):
        """
        Initialize the ZScoreScaler by loading the dataset and fitting the scaler to the training data.

        The scaler computes the mean and standard deviation from the training data, which is then used to 
        normalize the data during the `transform` operation.

        Args:
            dataset_name (str): The name of the dataset used to load the data.
            train_ratio (float): The ratio of the dataset to be used for training. The scaler is fitted on this portion of the data.
            norm_each_channel (bool): Flag indicating whether to normalize each channel separately. 
                If True, the mean and standard deviation are computed for each channel independently.
            rescale (bool): Flag indicating whether to apply rescaling after normalization. This flag is included for consistency with 
                the base class but is not directly used in Z-score normalization.
        """

        super().__init__(dataset_name, train_ratio, selected_channel, target_channel, norm_each_channel, rescale)
        self.target_channel = target_channel  # assuming normalization on the first channel
        self.selected_channel = selected_channel
        # load dataset description and data
        description_file_path = f'datasets/{dataset_name}/desc.json'
        with open(description_file_path, 'r') as f:
            description = json.load(f)
        data_file_path = f'datasets/{dataset_name}/data.dat'
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))
        _,_,C =data.shape
        self.all_channel = list(range(C))
        # split data into training set based on the train_ratio
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :, self.all_channel].copy()
        # DATA (T,N,D)
        # compute mean and standard deviation
        if norm_each_channel:
            self.mean = np.mean(train_data, axis=0, keepdims=True)
            self.std = np.std(train_data, axis=0, keepdims=True)
            self.std[self.std == 0] = 1.0  # prevent division by zero by setting std to 1 where it's 0
        else:
            self.mean = np.mean(train_data)
            self.std = np.std(train_data)
            if self.std == 0:
                self.std = 1.0  # prevent division by zero by setting std to 1 where it's 0

        self.mean, self.std = torch.tensor(self.mean), torch.tensor(self.std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Apply Z-score normalization to the input data.

        This method normalizes the input data using the mean and standard deviation computed from the training data. 
        The normalization is applied only to the specified `target_channel`.

        Args:
            input_data (torch.Tensor): The input data to be normalized.

        Returns:
            torch.Tensor: The normalized data with the same shape as the input.
        """

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)

        for i, ch in enumerate(self.all_channel[:-2]):

            input_data[...,ch] = (input_data[..., ch] - mean[..., i]) / std[..., i]

        return input_data

    def inverse_transform(self, input_data: torch.Tensor, flag: str) -> torch.Tensor:
        """
        Reverse the Z-score normalization to recover the original data scale.

        This method transforms the normalized data back to its original scale using the mean and standard deviation 
        computed from the training data. This is useful for interpreting model outputs or for further analysis in the original data scale.

        Args:
            input_data (torch.Tensor): The normalized data to be transformed back.
            flag (str): The flag indicating whether the data is the target datas or not.
        Returns:
            torch.Tensor: The data transformed back to its original scale.
        """

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        # Clone the input data to prevent in-place modification (which is not allowed in PyTorch)
        input_data = input_data.clone()
        if flag == 'target_channel':
            for i, ch in enumerate(self.target_channel):
                if ch != len(self.all_channel) - 1 and ch != len(self.all_channel) - 2:
                    input_data[..., i] = input_data[..., i] * std[..., ch] + mean[..., ch]

        if flag == 'selected_channel':
            for i, ch in enumerate(self.selected_channel):
                if ch != len(self.all_channel) - 1 and ch != len(self.all_channel) - 2:
                    input_data[..., i] = input_data[..., i] * std[..., ch] + mean[..., ch]


        return input_data
