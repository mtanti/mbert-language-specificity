'''
About file.
'''

import socket
import os
from typing import Mapping
import mufins
from mufins.common.time.time_utils import get_readable_timestamp
from mufins.common.file.csv_file import CsvFile
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException


#########################################
class AboutFile():
    '''
    An about file is a CSV file consisting of the following information:

    - The current date and time.
    - The program's version.
    - The computer's hostname.
    - The calling script's path.
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
        self.file: CsvFile = CsvFile(path)

    #########################################
    def init(
        self,
    ) -> bool:
        '''
        If it does not already exist, create the file with the about information.

        :return: Whether the file was created or not.
        '''
        if self.file.init(['key', 'value']):
            timestamp = get_readable_timestamp()
            self.file.append(['timestamp', timestamp])
            self.file.append(['version', mufins.__version__])
            self.file.append(['hostname', socket.gethostname()])
            self.file.append(['path', os.getcwd()])
            return True

        lines = self.file.read()
        if next(lines) != ['key', 'value']:
            raise IncompatibleExistingDataException('Existing file is incompatible.')
        for expected in 'timestamp version hostname path'.split(' '):
            if next(lines)[0] != expected:
                raise IncompatibleExistingDataException('Existing file is incompatible.')
        if next(lines, None) is not None:
            raise IncompatibleExistingDataException('Existing file is incompatible.')
        return False

    #########################################
    def read(
        self,
    ) -> Mapping[str, str]:
        '''
        Read the contents of the file.

        :return: A mapping of keys to values.
        '''
        return {row[0]: row[1] for row in self.file.read(skip_headings=True)}
