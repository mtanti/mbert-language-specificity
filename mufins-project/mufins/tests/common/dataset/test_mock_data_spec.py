'''
Unit test for Mock Data Spec class in dataset tests module.
'''

import unittest
import numpy as np
from mufins.tests.common.dataset.mock_data_spec import MockDataSpec


#########################################
class TestMockDataSpec(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the MockDataSpec class.
        '''
        spec = MockDataSpec()

        self.assertEqual(
            spec.get_field_specs(),
            {
                'a': ([3], np.int32),
                'b': ([1], np.int32),
            }
        )
        np.testing.assert_equal(
            spec.preprocess('1234'),
            {
                'a': np.array([1, 2, 3], np.int32),
                'b': np.array([4], np.int32),
            }
        )


#########################################
if __name__ == '__main__':
    unittest.main()
