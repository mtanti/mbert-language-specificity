'''
Adversarial abstract neural network model class.

An adversarial model has a part of it being a discriminator which is a classifier that is trained
to classify the existence of features in the model which should not exist.
The discriminator is trained on a special data set for this purpose.
The rest of the model is then trained to perform both the target task and to fool the discriminator
into performing badly, which requires the model to stop producing the undesirable features.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from mufins.common.model.model import Model


#########################################
class ModelAdversarial(Model):
    '''
    The base backend class for adversarial models.
    '''

    #########################################
    def __init__(
        self,
        net: nn.Module,
        opt_disc: optim.Optimizer,
        opt_main: optim.Optimizer,
        device: torch.device,
    ) -> None:
        '''
        Constructor.

        The neural network will be moved to the given device and set to non-training mode.

        :param net: The neural network.
        :param opt_disc: The training optimiser for the discriminator.
        :param opt_main: The training optimiser for the main part of the model.
        :param device: The GPU device name to use e.g. 'cuda:0'.
        '''
        super().__init__(net, [opt_disc, opt_main], device)
