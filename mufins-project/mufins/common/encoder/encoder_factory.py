'''
Encoder factory.
'''

from typing import Optional
from mufins.common.encoder.encoder_mbert import Encoder
from mufins.common.encoder.encoder_mbert import EncoderMbert
from mufins.common.encoder.encoder_mbert_trunc_cls import EncoderMbertTruncCls
from mufins.common.encoder.encoder_mbert_trunc_mean import EncoderMbertTruncMean
from mufins.common.encoder.encoder_mock import EncoderMock


#########################################
ENCODER_NAMES = {'mbert', 'mbert_trunc_cls', 'mbert_trunc_mean', 'mock_encoder'}


#########################################
def encoder_factory(
    encoder_name: str,
    layer_index: Optional[int],
) -> Encoder:
    '''
    Create an encoder instance from an encoder name.

    :param encoder_name: The name of the encoder. It can be one of the following:
        - 'mbert': Create an instance of `EncoderMbert`.
        - 'mbert_trunc_cls': Create an instance of `EncoderMbertTruncCls`.
        - 'mbert_trunc_mean': Create an instance of `EncoderMbertTruncMean`.
        - 'mock_encoder': Create an instance of `EncoderMock`.
    :param layer_index: The layer to use in the encoder.
    :return: The instantiated encoder.
    '''
    if encoder_name == 'mbert':
        return EncoderMbert()
    if encoder_name == 'mbert_trunc_cls':
        return EncoderMbertTruncCls(layer_index)
    if encoder_name == 'mbert_trunc_mean':
        return EncoderMbertTruncMean(layer_index)
    if encoder_name == 'mock_encoder':
        return EncoderMock(EncoderMbert.VECTOR_SIZE)
    raise ValueError('Unknown encoder name.')
