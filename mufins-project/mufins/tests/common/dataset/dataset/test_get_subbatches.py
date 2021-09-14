'''
Unit test for get_subbatches function in dataset module.
'''

import unittest
import numpy as np
from mufins.common.dataset.dataset import get_subbatches


#########################################
class TestGetSubbatches(unittest.TestCase):
    '''
    Unit test class.
    '''

    ########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the get_subbatches function.
        '''
        # pylint: disable=no-self-use
        np.testing.assert_equal(
            list(get_subbatches(
                {
                    'a': np.array([[1,2],[3,4],[5,6]], np.float32),
                    'b': np.array([[2,1],[4,3],[6,5]], np.float32),
                },
                2
            )),
            [
                {
                    'a': np.array([[1,2],[3,4]], np.float32),
                    'b': np.array([[2,1],[4,3]], np.float32),
                },
                {
                    'a': np.array([[5,6]], np.float32),
                    'b': np.array([[6,5]], np.float32),
                },
            ]
        )

        np.testing.assert_equal(
            list(get_subbatches(
                {
                    'a': np.array([[1,2],[3,4],[5,6]], np.float32),
                    'b': np.array([[2,1],[4,3],[6,5]], np.float32),
                },
                5
            )),
            [
                {
                    'a': np.array([[1,2],[3,4],[5,6]], np.float32),
                    'b': np.array([[2,1],[4,3],[6,5]], np.float32),
                },
            ]
        )


#########################################
if __name__ == '__main__':
    unittest.main()
