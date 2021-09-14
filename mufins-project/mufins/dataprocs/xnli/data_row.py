'''
Composite object of all the information needed to make a XNLI dataset
row.
'''

from typing import Optional


#########################################
class XNLIDataRow():
    '''
    Composite object with the raw data needed for a row in a XNLI
    dataset.
    '''
    # pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        lang: Optional[str],
        premise_text: str,
        hypothesis_text: str,
        label: Optional[str],
    ) -> None:
        '''
        Constructor.

        :param lang: The sentence's language.
        :param premise_text: The premise text.
        :param hypothesis_text: The hypothesis text.
        :param label: The label of the two texts.
        '''
        self.lang: Optional[str] = lang
        self.premise_text: str = premise_text
        self.hypothesis_text: str = hypothesis_text
        self.label: Optional[str] = label
