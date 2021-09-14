'''
CSV file abstraction.
'''

import csv
from typing import Sequence, Iterator
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException
from mufins.common.error.invalid_state import InvalidStateException


#########################################
class CsvFile():
    '''
    CSV file class.
    '''

    #########################################
    def __init__(
        self,
        path: str,
    ) -> None:
        '''
        Constructor.

        :param path: Path to the CSV file.
        '''
        self.path: str = path
        self.inited: bool = False
        self.headings: Sequence[str] = []

    #########################################
    def init(
        self,
        headings: Sequence[str],
        clear: bool = False,
    ) -> bool:
        '''
        Create the CSV file if it does not exist.

        :param headings: A list of titles for each column.
        :param clear: Whether to clear the contents of the file if it exists.
        :return: Whether the file was created or not.
        '''
        try:
            with open(self.path, 'x', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerow(headings)
                self.headings = headings
                self.inited = True
                return True
        except FileExistsError:
            try:
                with open(self.path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.reader(f, dialect='excel')
                    headings_found = next(reader)
            except StopIteration:
                raise IncompatibleExistingDataException('Existing file is incompatible.') from None
            if headings_found != list(headings):
                raise IncompatibleExistingDataException('Existing file is incompatible.') from None
            if clear:
                with open(self.path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, dialect='excel')
                    writer.writerow(headings)
            self.headings = headings
            self.inited = True
            return False

    #########################################
    def append(
        self,
        row: Sequence[object],
    ) -> None:
        '''
        Append a new row to the file.

        :param row: A list of objects that form the row in string form.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised file.')
        if len(row) != len(self.headings):
            raise ValueError('Incorrect number of fields.')
        with open(self.path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, dialect='excel')
            writer.writerow([str(field) for field in row])

    #########################################
    def read(
        self,
        skip_headings: bool = False,
    ) -> Iterator[Sequence[str]]:
        '''
        Read each row of the file.

        :param skip_headings: Whether to include the first row or not.
        :return: An iterator of rows consisting of lists of strings.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised file.')
        with open(self.path, 'r', encoding='utf-8', newline='') as f:
            reader = iter(csv.reader(f, dialect='excel'))
            if skip_headings:
                try:
                    next(reader)
                except StopIteration:
                    return
            for row in reader:
                yield row
