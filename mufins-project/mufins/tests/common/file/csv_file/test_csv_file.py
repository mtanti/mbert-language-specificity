'''
Unit test for CSV File class in file module.
'''

import os
import unittest
import tempfile
from mufins.common.file.csv_file import CsvFile
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException


#########################################
class TestCsvFile(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_init(
        self,
    ) -> None:
        '''
        Test the init method.
        '''
        with tempfile.TemporaryDirectory() as path:
            file = CsvFile(os.path.join(path, 'file.csv'))

            self.assertTrue(
                file.init(['head1', 'head2', 'head3'])
            )
            self.assertFalse(
                file.init(['head1', 'head2', 'head3'])
            )

            with self.assertRaises(IncompatibleExistingDataException):
                file.init(['head1', 'head2'])
            with open(os.path.join(path, 'file.csv'), 'w'):
                pass
            with self.assertRaises(IncompatibleExistingDataException):
                file.init(['head1', 'head2', 'head3'])

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the CsvFile class.
        '''
        with tempfile.TemporaryDirectory() as path:
            file = CsvFile(os.path.join(path, 'file.csv'))
            file.init(['head1', 'head2', 'head3'])

            file.append(['a', 1, False])
            file.append(['b', 2, True])

            with self.assertRaises(ValueError):
                file.append(['a', 1])
            with self.assertRaises(ValueError):
                file.append(['a', 1, False, 1])

            content = list(file.read())

            self.assertEqual(
                content,
                [
                    ['head1', 'head2', 'head3'],
                    ['a', '1', 'False'],
                    ['b', '2', 'True'],
                ],
            )

            content = list(file.read(skip_headings=True))

            self.assertEqual(
                content,
                [
                    ['a', '1', 'False'],
                    ['b', '2', 'True'],
                ],
            )

            file.init(['head1', 'head2', 'head3'], clear=True)
            self.assertEqual(
                list(file.read()),
                [
                    ['head1', 'head2', 'head3'],
                ],
            )

            file = CsvFile(os.path.join(path, 'file2.csv'))
            file.init(['head1', 'head2', 'head3'], clear=True)
            file.append(['a', 1, False])
            file.append(['b', 2, True])
            self.assertEqual(
                list(file.read()),
                [
                    ['head1', 'head2', 'head3'],
                    ['a', '1', 'False'],
                    ['b', '2', 'True'],
                ],
            )



#########################################
if __name__ == '__main__':
    unittest.main()
