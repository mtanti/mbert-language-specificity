'''
Tokeniser factory.
'''

from mufins.common.tokeniser.tokeniser_mbert import Tokeniser
from mufins.common.tokeniser.tokeniser_mbert import TokeniserMbert
from mufins.common.tokeniser.tokeniser_mock import TokeniserMock


#########################################
TOKENISER_NAMES = {'mbert'}


#########################################
def tokeniser_factory(
    tokeniser_name: str,
) -> Tokeniser:
    '''
    Create a tokeniser instance from a tokeniser name.

    :param tokeniser_name: The name of the tokeniser. It can be one of the following:
        - 'mbert': Create an instance of `TokeniserMbert`.
    :return: The instantiated tokeniser.
    '''
    if tokeniser_name == 'mbert':
        return TokeniserMbert()
    if tokeniser_name == 'mock':
        return TokeniserMock()
    raise ValueError('Unknown tokeniser name.')
