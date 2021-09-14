'''
Abstract base neural network model class.
'''

import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd


#########################################
class Model(ABC):
    '''
    The base backend class.
    '''

    #########################################
    @staticmethod
    def get_device(
        device_name: str,
    ) -> torch.device:
        '''
        Convert a device name into a PyTorch device object.

        If it does not exist, this will return a CPU device.

        :param device_name: The name of the device such as 'cpu', 'cuda:0', 'cuda:1', etc.
        :return: The torch device object.
        '''
        return torch.device(device_name if torch.cuda.is_available() else 'cpu')

    #########################################
    def __init__(
        self,
        net: nn.Module,
        opts: Sequence[optim.Optimizer],
        device: torch.device,
    ) -> None:
        '''
        Constructor.

        The neural network will be moved to the given device and set to non-training mode.

        :param net: The neural network.
        :param opts: The training optimisers that can be used.
        :param device: The GPU device name to use e.g. 'cuda:0'.
        '''
        self.net: nn.Module = net
        self.opts: Sequence[optim.Optimizer] = opts
        self.device: torch.device = device
        self.__curr_opt_index: int = 0

        self.net_changed()

    #########################################
    def net_changed(
        self,
    ) -> None:
        '''
        Stuff to do when the neural network structure has been updated.

        Will set training mode to false and move network to correct device.
        '''
        self.net.train(False)
        self.net.to(self.device)

    #########################################
    def save_state(
        self,
        path: str,
    ) -> None:
        '''
        Save the current state of the model parameters and optimiser.

        :param path: The path to the pickle file name to be saved.
        '''
        obj = {
            'net': self.net.state_dict(),
            'opts': [opt.state_dict() for opt in self.opts],
        }
        torch.save(obj, path)

    #########################################
    def load_state(
        self,
        path: str,
    ) -> None:
        '''
        Load the saved state of the model parameters and optimiser.

        :param path: The path to the pickle file name to be loaded.
        '''
        obj = torch.load(path, map_location=self.device)
        self.net.load_state_dict(obj['net'])
        for (opt, param) in zip(self.opts, obj['opts']):
            opt.load_state_dict(param)

    #########################################
    def save_params(
        self,
        path: str,
    ) -> None:
        '''
        Save the current parameters of the model.

        :param path: The path to the pickle file name to be saved.
        '''
        obj = self.net.state_dict()
        torch.save(obj, path)

    #########################################
    def load_params(
        self,
        path: str,
    ) -> None:
        '''
        Load the saved parameters of the model.

        :param path: The path to the pickle file name to be loaded.
        '''
        obj = torch.load(path, map_location=self.device)
        self.net.load_state_dict(obj)

    #########################################
    def get_curr_opt_index(
        self,
    ) -> int:
        '''
        Get the current optimiser index.

        :return: The index of the current optimiser.
        '''
        return self.__curr_opt_index

    #########################################
    def set_curr_opt_index(
        self,
        index: int,
    ) -> None:
        '''
        Set the current optimiser index.

        :param index: The index of the current optimiser.
        '''
        if not 0 <= index < len(self.opts):
            raise ValueError('Optimiser index must be between 0 and {}.'.format(len(self.opts)))

        self.__curr_opt_index = index

    #########################################
    @abstractmethod
    def _set_gradients(
        self,
        batch: Sequence[Mapping[str, np.ndarray]],
        batch_size: int,
    ) -> None:
        '''
        Set the gradients of the parameters with a single batch of data.

        :param batch: The batch of data.
        :param batch_size: The maximum amount of training items to process in one go.
        '''

    #########################################
    def batch_fit(
        self,
        batch: Sequence[Mapping[str, np.ndarray]],
        batch_size: int,
    ) -> None:
        '''
        Update the model parameters using the optimiser with a single batch of data.

        The neural network will be temporarily set to training mode and back to non-training mode.

        :param batch: The batch of data.
        :param batch_size: The maximum amount of training items to process in one go.
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with autograd.detect_anomaly():
                self.net.train(True)
                self.opts[self.__curr_opt_index].zero_grad()
                self._set_gradients(batch, batch_size)
                self.opts[self.__curr_opt_index].step()
                self.net.train(False)
