'''
Abstract neural network encoder class for encoding text into vectors.
'''

from abc import ABC, abstractmethod
from typing import Tuple, Iterator
import torch
import torch.nn as nn


#########################################
class Encoder(ABC, nn.Module):
    '''
    Abstract encoder class.
    '''

    #########################################
    @abstractmethod
    def get_parameters(
        self,
        with_embeddings: bool = True,
    ) -> Iterator[nn.Parameter]:
        '''
        Get the parameters of the model.

        :param with_embeddings: Whether to include the embedding parameter or not.
        :return: The parameters.
        '''

    #########################################
    @abstractmethod
    def get_embedding_parameter(
        self,
    ) -> nn.Parameter:
        '''
        Get the embedding parameter only.

        :return: The embedding parameter.
        '''

    #########################################
    @abstractmethod
    def forward(
        self,
        token_indexes: torch.Tensor,
        token_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass.

        :param token_indexes: A tensor of token indexes originating from a tokeniser.
        :param token_masks: A tensor of booleans saying which tokens are to be used from
            `token_indexes`.
        :return: A pair consisting of a tensor of encoded tokens of shape (text, token, vector)
            and a tensor of the encoded text of shape (text, vector).
        '''

    #########################################
    def encode_tokens(
        self,
        token_indexes: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Encode a batch of tokens.

        :param token_indexes: A tensor of token indexes originating from a tokeniser.
        :param token_masks: A tensor of booleans saying which tokens are to be used from
            `token_indexes`.
        :return: A tensor of encoded tokens of shape (text, token, vector).
        '''
        return self(token_indexes, token_masks)[0]

    #########################################
    def encode_texts(
        self,
        token_indexes: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Encode a batch of texts.

        :param token_indexes: A tensor of token indexes originating from a tokeniser.
        :param token_masks: A tensor of booleans saying which tokens are to be used from
            `token_indexes`.
        :return: A tensor of encoded overall text of shape (text, vector).
        '''
        return self(token_indexes, token_masks)[1]
