'''
Unit test for Version File class in file module.
'''

import os
import unittest
import tempfile
from mufins.common.file.version_file import VersionFile
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException


#########################################
class TestVersionFile(unittest.TestCase):
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
            file = VersionFile(os.path.join(path, 'file.txt'))

            self.assertTrue(
                file.init(1)
            )
            self.assertFalse(
                file.init(1)
            )

            with self.assertRaises(IncompatibleExistingDataException):
                file.init(2)

            with open(os.path.join(path, 'file.txt'), 'w'):
                pass
            with self.assertRaises(IncompatibleExistingDataException):
                file.init(1)

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the VersionFile class.
        '''
        with tempfile.TemporaryDirectory() as path:
            file = VersionFile(os.path.join(path, 'file.txt'))
            self.assertTrue(
                file.init(1)
            )
            self.assertEqual(file.read(), 1)

            file.init()
            self.assertEqual(file.read(), 1)


#########################################
if __name__ == '__main__':
    unittest.main()
