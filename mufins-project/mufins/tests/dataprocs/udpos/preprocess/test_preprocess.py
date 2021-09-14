'''
Unit test for UDPOS Preprocess function in udpos module.
'''

import sys
import shutil
import unittest
import os
import tempfile
import argparse
import mufins
from mufins.dataprocs.udpos.preprocess import udpos_preprocess


#########################################
get_path = False # pylint: disable=invalid-name
verbose = False # pylint: disable=invalid-name


#########################################
class TestUDPOSPreprocess(unittest.TestCase):
    '''
    Unit test class.
    '''
    # pylint: disable=no-self-use

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the udpos_preprocess function.
        '''
        with tempfile.TemporaryDirectory() as path:
            shutil.copytree(
                os.path.join(
                    mufins.path, 'tests', 'dataprocs', 'udpos', 'mock_dataset',
                ),
                os.path.join(path, 'src'),
            )

            udpos_preprocess(
                src_path=os.path.join(path, 'src'),
                dst_path=os.path.join(path, 'dst'),
                tokeniser_name='mock',
                train_fraction=0.6,
                val_fraction=0.4,
                max_num_tokens=7,
                seed=0,
                verbose=verbose,
            )

            if get_path:
                print(path)
                input()
                return


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--get_path',
        required=False,
        choices=['yes', 'no'],
        default='no',
        help='Whether to display the temporary path and not actually test.'
    )
    parser.add_argument(
        '--verbose',
        required=False,
        choices=['yes', 'no'],
        default='no',
        help='Whether to display the output.'
    )
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    get_path = args.get_path == 'yes'
    verbose = args.verbose == 'yes'
    sys.argv[1:] = args.unittest_args
    unittest.main()
