'''
Mock model class.
'''

import math
from typing import cast, Sequence, Mapping
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mufins.common.model.model_standard import ModelStandard


#########################################
class MockModel(ModelStandard):
    '''
    A mock implemenation of the Model class that implements a single layer neural network
    with no activation function. It maps two numbers into one number and the weights and biases
    are initialised to all ones. The target is to make the outputs all ones.

    The optimiser can be chosen between SGD and Adam.
    '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        opt_name: str,
    ) -> None:
        '''
        Constructor.

        :param opt_name: Optimiser name can be 'sgd' or 'adam'.
        '''
        device = ModelStandard.get_device('cpu')

        #########################################
        class Net(nn.Module):
            '''
            The neural network definition.
            '''

            #########################################
            def __init__(
                self,
            ) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 1)
                self.linear.weight.data = torch.ones_like(self.linear.weight.data)
                self.linear.bias.data = torch.ones_like(self.linear.bias.data)

            #########################################
            def forward(
                self,
                data_in: np.ndarray,
            ) -> torch.Tensor:
                data_in_tensor = torch.Tensor(data_in, device=device)
                return self.linear(data_in_tensor)

        net = Net()

        if opt_name == 'sgd':
            opt = cast(optim.Optimizer, optim.SGD(net.parameters(), 0.1))
        elif opt_name == 'adam':
            opt = cast(optim.Optimizer, optim.Adam(net.parameters()))
        else:
            raise ValueError()

        super().__init__(net, [opt], device)

    #########################################
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
        batch_ = batch[0]['x']
        for i in range(math.ceil(batch_.shape[0]/batch_size)):
            outputs = self.net(batch_[i*batch_size:(i+1)*batch_size])
            loss = torch.sum((outputs - 1.0)**2)/batch_.shape[0]
            loss.backward()
