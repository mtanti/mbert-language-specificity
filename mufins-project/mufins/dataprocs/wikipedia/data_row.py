'''
Composite object of all the information needed to make a Wikipedia dataset row.
'''

from typing import Optional


#########################################
class WikipediaDataRow():
    '''
    Composite object with the raw data needed for a row in a Wikipedia dataset.
    '''
    # pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        text: str,
        lang: Optional[str],
    ) -> None:
        '''
        Constructor.

        :param text: The sentence.
        :param lang: The sentence's language.
        '''
        self.text: str = text
        self.lang: Optional[str] = lang
