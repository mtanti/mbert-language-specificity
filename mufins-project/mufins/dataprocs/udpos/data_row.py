'''
Composite object of all the information needed to make a UDPOS dataset
row.
'''

from typing import Sequence, Optional


#########################################
class UDPOSDataRow():
    '''
    Composite object with the raw data needed for a row in a UDPOS
    dataset.
    '''
    # pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        words: Sequence[str],
        lang: Optional[str],
        labels: Optional[Sequence[str]],
    ) -> None:
        '''
        Constructor.

        :param words: A sequence of words.
        :param lang: The sentence's language.
        :param labels: A sequence of labels for each token.
        '''
        self.words: Sequence[str] = words
        self.lang: Optional[str] = lang
        self.labels: Optional[Sequence[str]] = labels
