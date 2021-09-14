'''
Unit test for DatasetFile class in dataset module.
'''

import os
import unittest
import tempfile
from mufins.common.dataset.dataset_file import DatasetFile
from mufins.tests.common.dataset.mock_data_spec import MockDataSpec
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException
from mufins.common.error.invalid_state import InvalidStateException


#########################################
class TestDatasetFile(unittest.TestCase):
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
        spec = MockDataSpec()

        with tempfile.TemporaryDirectory() as path:
            dset = DatasetFile[str](
                os.path.join(path, 'test.hdf'),
                spec,
            )

            self.assertTrue(dset.init(3))
            self.assertFalse(dset.init(3))

            with self.assertRaises(IncompatibleExistingDataException):
                dset.init(10)

            with open(os.path.join(path, 'test.hdf'), 'w'):
                pass
            with self.assertRaises(IncompatibleExistingDataException):
                dset.init(3)

    ########################################
    def test_readwrite(
        self,
    ) -> None:
        '''
        Test the Dataset class when it is read from and modified.
        '''
        spec = MockDataSpec()

        with tempfile.TemporaryDirectory() as path:
            dset = DatasetFile[str](
                os.path.join(path, 'test.hdf'),
                spec,
            )
            dset.init(3)
            dset.load(as_readonly=False)

            dset.set_row(0, {'a': [0,1,2], 'b':[10]})
            dset.set_row(1, {'a': [3,4,5], 'b':[11]})
            dset.set_field(2, 'a', [6,7,8])
            dset.set_field(2, 'b', [12])

            dset.close()
            with self.assertRaises(InvalidStateException):
                dset.get_row(0)
            with self.assertRaises(InvalidStateException):
                dset.get_field(0, 'a')

            dset.init(3)
            dset.load(as_readonly=True)
            with self.assertRaises(InvalidStateException):
                dset.set_row(0, {'a': [0,0,0], 'b': [0]})
            with self.assertRaises(InvalidStateException):
                dset.set_field(0, 'a', [0,0,0])

            self.assertEqual(
                {name: value.tolist() for (name, value) in dset.get_row(0).items()},
                {'a': [0,1,2], 'b': [10]},
            )
            self.assertEqual(
                {name: value.tolist() for (name, value) in dset.get_row(1).items()},
                {'a': [3,4,5], 'b': [11]},
            )
            self.assertEqual(
                {'a': dset.get_field(2, 'a').tolist(), 'b': dset.get_field(2, 'b').tolist()},
                {'a': [6,7,8], 'b': [12]},
            )

            self.assertEqual(
                list(dset.get_data(
                    batch_size=2,
                    value_filter = lambda i, x: i != 1,
                    value_mapper = lambda i, x: x['a'].tolist(),
                )),
                [
                    [0, 1, 2],
                    [6, 7, 8],
                ]
            )

            dset.close()


#########################################
if __name__ == '__main__':
    unittest.main()
