'''
Class for tokenising text using a mock tokeniser.
'''

import re
from typing import Optional, Sequence, Tuple
from mufins.common.tokeniser.tokeniser import Tokeniser


#########################################
class TokeniserMock(Tokeniser):
    '''
    Mock tokeniser.
    '''

    VOCAB = (
        '<PAD> <SEP> <CLS> <UNK> ! ##W ##a ##d ##e ##g' # 0-9
        + ' ##i ##k ##l ##n ##o ##r ##s ##t ##y .' # 10-19
        + ' A Abbiam Aud BM Ch Giann H Hi I Joh' # 20-29
        + ' Le M McDonald No Pau Paul S Sh Th W' # 30-39
        + ' Wha a abbai al am and arrive at barke ben' # 40-49
        + ' bo bough bran buff ca can compani compr do dran' # 50-59
        + ' e fee form giochar goo guidar h ha hat hate' # 60-69
        + ' husban i l learnin like lov m ma macchin mal' # 70-79
        + ' mangiat marit mi mogli nam nic nigh nott nuotar odi' # 80-89
        + ' ou pe piac pizz pla ragazz seem sembr sent sic' # 90-99
        + ' sill slep sol st stran studiand studyin swimmin t tutt' # 100-109
        + ' un unwel uom vend vint wate weir wel wen wif' # 110-119
        + ' wo' # 120-129
    ).split(' ')
    TOKEN2INDEX = {t: i for (i, t) in enumerate(VOCAB)}

    #########################################
    @staticmethod
    def tokenise(
        text: str,
    ) -> Sequence[str]:
        '''
        Tokenise a text.

        Contiguous sequences of letters are always grouped into a single token, contiguous sequences
        spaces are grouped into another token and contiguous sequences of anything else are grouped
        into another token. Spaces are then ignored, and tokens are split into first letter and
        the rest of the letters with '##' in front of the second part.

        :param text: The text to tokenise.
        :return: The sequence of tokens.
        '''
        return [
            split_token
            for token in re.findall(r'[a-zA-Z]+| +|[^ a-zA-Z]+', text)
            if token != ' '
            for split_token in (
                [token[:-1], '##'+token[-1]] if len(token) > 1 else [token]
            )
        ]

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

    #########################################
    def get_pad_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get pad control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return ('<PAD>', 0)

    #########################################
    def get_sep_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get seperator control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return ('<SEP>', 1)

    #########################################
    def get_cls_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get class control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return ('<CLS>', 2)

    #########################################
    def get_unk_token(
        self,
    ) -> Tuple[str, int]:
        '''
        Get unknown control token.

        :return: A pair consisting of the string token version and the index.
        '''
        return ('<UNK>', 3)

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
        return True

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
        tokens = TokeniserMock.tokenise(text)
        if max_len is not None and len(tokens) + 2 > max_len:
            tokens = tokens[:max_len - 2]
        return (
            [2]
            + [TokeniserMock.TOKEN2INDEX.get(token, 3) for token in tokens] + [1]
        )

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
        tokens1 = TokeniserMock.tokenise(text1)
        tokens2 = TokeniserMock.tokenise(text2)
        if max_len is not None and len(tokens1) + len(tokens2) + 3 > max_len:
            tokens1 = tokens1[:(max_len - 3)//2]
            tokens2 = tokens2[:(max_len - 3)//2 + (max_len - 3)%2]
        return (
            [2]
            + [TokeniserMock.TOKEN2INDEX.get(token, 3) for token in tokens1] + [1]
            + [TokeniserMock.TOKEN2INDEX.get(token, 3) for token in tokens2] + [1]
        )

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
        tokens = [TokeniserMock.VOCAB[index] for index in indexes]
        return (
            [token if not token.startswith('##') else token[2:] for token in tokens],
            [not token.startswith('##') for token in tokens],
        )
