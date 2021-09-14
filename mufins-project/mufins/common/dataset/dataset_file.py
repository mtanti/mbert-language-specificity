'''
Dataset class for storing and using datasets on disk.
'''

import os
from typing import TypeVar, Optional
import h5py
import numpy as np
from mufins.common.dataset.data_spec import DataSpec
from mufins.common.dataset.dataset import Dataset
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException
from mufins.common.error.invalid_state import InvalidStateException


#########################################
T = TypeVar('T')

class DatasetFile(Dataset[T]):
    '''
    Dataset File class.

    T is a class for composite objects of raw data (such as texts or images).
    This object is expected to contain all the information necessary to
    produce an entire line of fields in the dataset, including target values.
    '''

    #########################################
    def __init__(
        self,
        path: str,
        spec: DataSpec[T],
    ) -> None:
        '''
        Constructor.

        :param path: The path of the dataset HDF5 file.
        :param spec: The data specification object that describes the data
            to store in the dataset.
        '''
        super().__init__(spec)
        self.path: str = path

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
        file_exists = os.path.isfile(self.path)
        if file_exists:
            try:
                with h5py.File(self.path, 'r') as f:
                    if 'size' not in f.attrs:
                        raise IncompatibleExistingDataException(
                            'Existing file is incompatible.'
                        ) from None
                    if self.size == 0:
                        self.size = f.attrs['size']
                    if size is None:
                        size = self.size
                    elif self.size != size:
                        raise IncompatibleExistingDataException(
                            'Existing file is incompatible.'
                        ) from None
                    field_specs = self.spec.get_field_specs()
                    if set(f.keys()) != set(field_specs.keys()):
                        raise IncompatibleExistingDataException(
                            'Existing file is incompatible.'
                        ) from None
                    for (name, (shape, dtype)) in field_specs.items():
                        if (
                            list(f[name].shape) != [size]+list(shape)
                            or str(f[name].dtype) != str(np.dtype(dtype))
                        ):
                            raise IncompatibleExistingDataException(
                                'Existing file is incompatible.'
                            ) from None
                    return False
            except OSError as ex:
                if str(ex).startswith('Unable to open file'):
                    raise IncompatibleExistingDataException(
                        'Existing file is incompatible.'
                    ) from None
                raise ex
        else:
            if size is None:
                raise ValueError('Size must be given when data is to be created.')
            if self.size > 0 and size != self.size:
                raise IncompatibleExistingDataException(
                    'Existing file is incompatible.'
                ) from None
            with h5py.File(self.path, 'x') as f:
                f.attrs['size'] = size
                field_specs = self.spec.get_field_specs()
                for (name, (shape, dtype)) in field_specs.items():
                    f.create_dataset(
                        name,
                        [size]+list(shape),
                        dtype=dtype,
                        chunks=None,
                    )
            self.size = size
            return True

    #########################################
    def load(
        self,
        as_readonly: bool = False,
    ) -> None:
        '''
        Load the dataset.

        :param as_readonly: Whether the dataset can be modified or not.
            Only effective for saved datasets.
        '''
        if self.size == 0:
            raise InvalidStateException('Cannot load before initialising.')
        self.data = h5py.File(
            self.path,
            'r' if as_readonly else 'r+',
        )
        self.readonly = as_readonly

    #########################################
    def close(
        self,
    ) -> None:
        '''
        Close the dataset.
        '''
        if isinstance(self.data, DatasetFile):
            self.data.close()
        self.data = None
        self.size = 0
        self.readonly = False
