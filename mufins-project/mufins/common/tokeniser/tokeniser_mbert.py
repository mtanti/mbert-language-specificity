'''
Class for tokenising text using m-BERT's tokeniser.
'''

from typing import Optional, Sequence, Tuple
import transformers
from mufins.common.tokeniser.tokeniser import Tokeniser


#########################################
class TokeniserMbert(Tokeniser):
    '''
    m-BERT tokeniser.
    '''

    MAX_LEN = 512

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Constructor.
        '''
        super().__init__(
            noncontent_front=1,
            noncontent_back=1
        )

        self.tokeniser: transformers.BertTokenizer = transformers.BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased'
        )

    #########################################
    def get_pad_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get pad control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return (self.tokeniser.pad_token, self.tokeniser.pad_token_id)

    #########################################
    def get_sep_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get seperator control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return (self.tokeniser.sep_token, self.tokeniser.sep_token_id)

    #########################################
    def get_cls_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get class control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return (self.tokeniser.cls_token, self.tokeniser.cls_token_id)

    #########################################
    def get_unk_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get unknown control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return (self.tokeniser.unk_token, self.tokeniser.unk_token_id)

    #########################################
    def all_known_tokens(
        self,
        text: str,
    ) -> bool:
        '''
        Check if any of the tokens in the text would be encoded as an unknown token.

        :param text: The text to check.
        :return: Whether there will be an unknown token.
        '''
        return self.tokeniser.unk_token_id not in self.indexify_text(text)

    #########################################
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
        return self.tokeniser(
            text,
            truncation=True,
            max_length=max_len,
        )['input_ids']

    #########################################
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
        return self.tokeniser(
            text1,
            text2,
            truncation=True,
            max_length=max_len,
        )['input_ids']

    #########################################
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
        tokens = self.tokeniser.convert_ids_to_tokens(indexes)
        return (
            [token if not token.startswith('##') else token[2:] for token in tokens],
            [not token.startswith('##') for token in tokens],
        )
