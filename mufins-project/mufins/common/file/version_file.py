'''
Version file.
'''

import os
from typing import Optional
from mufins.common.error.invalid_state import InvalidStateException
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException


#########################################
class VersionFile():
    '''
    A file for storing a version number.
    '''

    #########################################
    def __init__(
        self,
        path: str,
    ) -> None:
        '''
        Constructor.

        :param path: The path to the file.
        '''
        self.path: str = path
        self.inited: bool = False

    #########################################
    def init(
        self,
        version: Optional[int] = None,
    ) -> bool:
        '''
        If it does not already exist, create the file with the version information.

        :param version: The version to save in the file.
        :return: Whether the file was created or not.
        '''
        if not os.path.isfile(self.path):
            if version is None:
                raise ValueError('Version must be given when file does not exist.')
            with open(self.path, 'x', encoding='utf-8') as f:
                print(version, file=f)
            self.inited = True
            return True

        with open(self.path, 'r', encoding='utf-8') as f:
            try:
                found_version = int(f.read().strip())
            except ValueError:
                raise IncompatibleExistingDataException(
                    'Existing file is incompatible.'
                ) from None
            if found_version < 1:
                raise IncompatibleExistingDataException('Existing file is incompatible.')
            if version is not None:
                if version != found_version:
                    raise IncompatibleExistingDataException('Existing file is incompatible.')
        self.inited = True
        return False

    #########################################
    def read(
        self,
    ) -> int:
        '''
        Get the version in the file.

        :return: The version number.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot read an uninitialised file.')

        with open(self.path, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
