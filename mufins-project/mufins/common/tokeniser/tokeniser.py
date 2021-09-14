'''
Abstract tokeniser class for tokenising text.
'''

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple


#########################################
class Tokeniser(ABC):
    '''
    Abstract tokeniser class.
    '''

    #########################################
    def __init__(
        self,
        noncontent_front: int,
        noncontent_back: int,
    ) -> None:
        '''
        Constructor.

        :param noncontent_front: The number of control tokens added to the front of an indexified
            text.
        :param noncontent_back: The number of control tokens added to the back of an indexified
            text.
        '''
        self.noncontent_front: int = noncontent_front
        self.noncontent_back: int = noncontent_back
        self.noncontent_total: int = noncontent_front + noncontent_back
        self.content_slice: slice = slice(noncontent_front, -noncontent_back)

    #########################################
    @abstractmethod
    def get_pad_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get pad control token.

        :return: A pair consisting of the string token version and the index.
        '''

    #########################################
    @abstractmethod
    def get_sep_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get seperator control token.

        :return: A pair consisting of the string token version and the index.
        '''

    #########################################
    @abstractmethod
    def get_cls_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get class control token.

        :return: A pair consisting of the string token version and the index.
        '''

    #########################################
    @abstractmethod
    def get_unk_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get unknown control token.

        :return: A pair consisting of the string token version and the index.
        '''

    #########################################
    @abstractmethod
    def all_known_tokens(
        self,
        text: str,
    ) -> bool:
        '''
        Check if any of the tokens in the text would be encoded as an unknown token.

        :param text: The text to check.
        :return: Whether there will be an unknown token.
        '''

    #########################################
    @abstractmethod
    def indexify_text(
        self,
        text: str,
        max_len: Optional[int] = None,
    ) -> Sequence[int]:
        '''
        Indexify a single text.

        :param text: The text to indexify.
        :param max_len: The maximum number of tokens including control tokens.
        :return: A sequence of indexes.
        '''

    #########################################
    @abstractmethod
    def indexify_text_pair(
        self,
        text1: str,
        text2: str,
        max_len: Optional[int] = None,
    ) -> Sequence[int]:
        '''
        Indexify a pair of texts.

        :param text1: The first text to indexify.
        :param text2: The second text to indexify.
        :param max_len: The combined maximum number of tokens including control tokens.
        :return: A sequence of indexes.
        '''

    #########################################
    @abstractmethod
    def textify_indexes(
        self,
        indexes: Sequence[int],
    ) -> Tuple[Sequence[str], Sequence[bool]]:
        '''
        Replace indexes with corresponding string tokens without inner-token markers.

        :param indexes: The sequence of indexes.
        :return: A pair consisting of the sequence of string tokens and a corresponding sequence
            of booleans indicating if the token is the first subtoken of a word or not.
        '''
