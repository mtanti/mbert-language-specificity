'''
Class for encoding text using truncated m-BERT with mean of tokens being used to encode text.

The mean excludes the CLS token.
'''

from typing import Tuple, Iterator, Optional
import transformers
import torch
import torch.nn as nn
from mufins.common.encoder.encoder import Encoder


#########################################
class EncoderMbertTruncMean(Encoder):
    '''
    Truncated m-BERT encoder with text encoder using mean method.
    '''

    VECTOR_SIZE = 768
    VOCAB_SIZE = 119547

    #########################################
    def __init__(
        self,
        layer_index: Optional[int] = None,
    ) -> None:
        '''
        Constructor.

        :param layer_index: The layer to use (0-12, where 0 is the word embeddings with positional
            embeddings).
            If None then last layer is used.
        '''
        super().__init__()

        self.layer_index: int = layer_index if layer_index is not None else 12
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
        (_, _, hidden_layers) = self.model(
            token_indexes, attention_mask=token_masks, output_hidden_states=True
        )

        return (
            hidden_layers[self.layer_index],
            hidden_layers[self.layer_index][:, 1:, :].mean(dim=1), # CLS token is at index 0.
        )
