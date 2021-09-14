'''
Unit test for About File class in file module.
'''

import os
import unittest
import tempfile
from mufins.common.file.about_file import AboutFile
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException


#########################################
class TestAboutFile(unittest.TestCase):
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
            file = AboutFile(os.path.join(path, 'file.csv'))

            self.assertTrue(
                file.init()
            )
            self.assertFalse(
                file.init()
            )

            with open(os.path.join(path, 'file.csv'), 'w'):
                pass
            with self.assertRaises(IncompatibleExistingDataException):
                file.init()

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the AboutFile class.
        '''
        with tempfile.TemporaryDirectory() as path:
            file = AboutFile(os.path.join(path, 'file.csv'))
            self.assertTrue(
                file.init()
            )
            content = file.read()
            self.assertEqual(
                set(content.keys()),
                {
                    'timestamp',
                    'version',
                    'hostname',
                    'path',
                }
            )


#########################################
if __name__ == '__main__':
    unittest.main()
