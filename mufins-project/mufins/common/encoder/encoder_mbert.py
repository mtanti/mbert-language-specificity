'''
Class for encoding text using m-BERT.
'''

from typing import Tuple, Iterator
import transformers
import torch
import torch.nn as nn
from mufins.common.encoder.encoder import Encoder


#########################################
class EncoderMbert(Encoder):
    '''
    m-BERT encoder.
    '''

    VECTOR_SIZE = 768
    VOCAB_SIZE = 119547

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Constructor.
        '''
        super().__init__()

        self.model: transformers.BertModel = transformers.BertModel.from_pretrained(
            'bert-base-multilingual-cased'
        )

    #########################################
    def get_parameters(
        self,
        with_embeddings: bool = True,
    ) -> Iterator[nn.Parameter]:
        '''
        Get the parameters of the model.

        :param with_embeddings: Whether to include the embedding parameter or not.
        :return: The parameters.
        '''
        params = self.model.parameters()
        if not with_embeddings:
            next(params)
        return params

    #########################################
    def get_embedding_parameter(
        self,
    ) -> nn.Parameter:
        '''
        Get the embedding parameter only.

        :return: The embedding parameter.
        '''
        return next(self.model.parameters())

    #########################################
    def forward(
        self,
        token_indexes: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass.

        :param token_indexes: A tensor of token indexes originating from a tokeniser.
        :param token_masks: A tensor of booleans saying which tokens are to be used from
            `token_indexes`.
        :return: A pair consisting of a tensor of encoded tokens of shape (text, token, vector)
            and a tensor of the encoded text of shape (text, vector).
        '''
        return self.model(token_indexes, attention_mask=token_masks)
