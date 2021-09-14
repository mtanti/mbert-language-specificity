'''
Data specification class for specifying datasets and how to convert data into
a suitable dataset format.
'''

from abc import ABC, abstractmethod
from typing import Generic, Mapping, Sequence, Tuple, TypeVar, Union
import numpy as np


#########################################
T = TypeVar('T')

class DataSpec(ABC, Generic[T]):
    '''
    Data specification class.

    T is a generic for composite objects of raw data (such as texts or images).
    This object is expected to contain all the information necessary to
    produce an entire line of fields in the dataset, including target values.
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Empty constructor.
        '''

    #########################################
    @abstractmethod
    def get_field_specs(
        self,
    ) -> Mapping[str, Tuple[Sequence[int], Union[np.dtype, str]]]:
        '''
        Get a mapping of name - array specifications that are expected in the
        dataset.

        :return: The field specifications.
        '''

    #########################################
    @abstractmethod
    def preprocess(
        self,
        raw: T,
    ) -> Mapping[str, np.ndarray]:
        '''
        Convert raw data into preprocessed numeric arrays, one for each dataset
        field.

        :param raw: A composite object of raw data (such as texts or images).
        :return: A mapping from field names to preprocessed data.
            Note that the batch dimension should not be included in the arrays.
        '''
