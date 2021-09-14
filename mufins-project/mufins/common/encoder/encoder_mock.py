'''
Class for encoding text using mock encoder for quick testing.
'''

from typing import Iterator, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mufins.common.encoder.encoder import Encoder
from mufins.common.tokeniser.tokeniser_mock import TokeniserMock
from mufins.common.random.random_number_generator import RandomNumberGenerator


#########################################
class EncoderMock(Encoder):
    '''
    Mock encoder.
    '''

    #########################################
    def __init__(
        self,
        vector_size: int,
    ) -> None:
        '''
        Constructor.

        :param vector_size: The size of the token vectors and the text vector.
        '''
        super().__init__()

        self.vector_size: int = vector_size

        rng = RandomNumberGenerator(0)

        self.embeddings: nn.Parameter = nn.Parameter(torch.tensor(
            rng.array_normal(0.0, 0.01, (len(TokeniserMock.VOCAB), vector_size)),
            dtype=torch.float32,
        ))

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
        if with_embeddings:
            return iter([self.embeddings])
        return iter([])

    #########################################
    def get_embedding_parameter(
        self,
    ) -> nn.Parameter:
        '''
        Get the embedding parameter only.

        :return: The embedding parameter.
        '''
        return self.embeddings

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
        token_enc = F.embedding(token_indexes, self.embeddings)
        mask = token_masks.reshape(token_masks.shape+(1,)).repeat(1, 1, self.vector_size)

        return (
            token_enc,
            torch.sum(token_enc*mask, dim=1)/token_masks.sum(dim=1),
        )
