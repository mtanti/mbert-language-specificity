'''
Dataset class for storing and using datasets in memory`.
'''

import typing
from typing import Dict, TypeVar, Optional
import numpy as np
from mufins.common.dataset.dataset import Dataset
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException
from mufins.common.error.invalid_state import InvalidStateException


#########################################
T = TypeVar('T')

class DatasetMemory(Dataset[T]):
    '''
    Dataset Memory class.

    T is a class for composite objects of raw data (such as texts or images).
    This object is expected to contain all the information necessary to
    produce an entire line of fields in the dataset, including target values.
    '''

    #########################################
    def init(
        self,
        size: Optional[int] = None,
    ) -> bool:
        '''
        Prepare to load the dataset and create it if it does not exist.

        :param size: The number of rows to reserve in the dataset.
            If None then the data is expected to already exist.
        :return: Whether the file was created or not.
        '''
        if self.data is not None:
            if size is None:
                size = self.size
            elif size != self.size:
                raise IncompatibleExistingDataException(
                    'Existing array is incompatible.'
                ) from None
        else:
            if size is None:
                raise ValueError('Size must be given when data is to be created.')
            if self.size > 0 and size != self.size:
                raise IncompatibleExistingDataException(
                    'Existing array is incompatible.'
                ) from None
            self.size = size
        return False

    #########################################
    def load(
        self,
        as_readonly: bool = False, # pylint: disable=unused-argument
    ) -> None:
        '''
        Load the dataset.

        :param as_readonly: Whether the dataset can be modified or not.
            Only effective for saved datasets.
        '''
        if self.size == 0:
            raise InvalidStateException('Cannot load before initialising.')
        field_specs = self.spec.get_field_specs()
        self.data = typing.cast(Dict[str, 'np.ndarray'], dict())
        for (name, (shape, dtype)) in field_specs.items():
            self.data[name] = np.empty(
                [self.size]+list(shape),
                dtype=dtype,
            )
        self.readonly = as_readonly

    #########################################
    def close(
        self,
    ) -> None:
        '''
        Close the dataset.
        '''
        self.data = None
        self.size = 0
        self.readonly = False
