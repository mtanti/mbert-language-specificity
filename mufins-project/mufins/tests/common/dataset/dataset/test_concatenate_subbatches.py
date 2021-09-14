'''
Unit test for concatenate_subbatches function in dataset module.
'''

import unittest
import numpy as np
from mufins.common.dataset.dataset import concatenate_subbatches


#########################################
class TestConcatenateSubbatches(unittest.TestCase):
    '''
    Unit test class.
    '''

    ########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the concatenate_subbatches function.
        '''
        # pylint: disable=no-self-use
        np.testing.assert_equal(
            concatenate_subbatches(
                [
                    {
                        'a': np.array([1,2], np.float32),
                        'b': np.array(1, np.float32),
                    },
                    {
                        'a': np.array([3,4], np.float32),
                        'b': np.array(2, np.float32),
                    },
                    {
                        'a': np.array([5,6], np.float32),
                        'b': np.array(3, np.float32),
                    },
                ]
            ),
            {
                'a': np.array([[1,2],[3,4],[5,6]], np.float32),
                'b': np.array([1,2,3], np.float32),
            }
        )

        np.testing.assert_equal(
            concatenate_subbatches(
                [
                    {
                        'a': np.array([1,2], np.float32),
                        'b': np.array(1, np.float32),
                    },
                ]
            ),
            {
                'a': np.array([[1,2]], np.float32),
                'b': np.array([1], np.float32),
            },
        )


#########################################
if __name__ == '__main__':
    unittest.main()
