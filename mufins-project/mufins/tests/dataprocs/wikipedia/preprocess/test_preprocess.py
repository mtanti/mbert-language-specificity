'''
Unit test for Wikipedia Preprocess function in wikipedia module.
'''

import sys
import shutil
import unittest
import os
import tempfile
import argparse
import mufins
from mufins.dataprocs.wikipedia.preprocess import wikipedia_preprocess


#########################################
get_path = False # pylint: disable=invalid-name
verbose = False # pylint: disable=invalid-name


#########################################
class TestWikipediaPreprocess(unittest.TestCase):
    '''
    Unit test class.
    '''
    # pylint: disable=no-self-use

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the wikipedia_preprocess function.
        '''
        with tempfile.TemporaryDirectory() as path:
            shutil.copytree(
                os.path.join(
                    mufins.path, 'tests', 'dataprocs', 'wikipedia', 'mock_dataset',
                ),
                os.path.join(path, 'src'),
            )

            wikipedia_preprocess(
                src_path=os.path.join(path, 'src'),
                dst_path=os.path.join(path, 'dst'),
                tokeniser_name='mock',
                train_fraction=0.4,
                val_fraction=0.2,
                dev_fraction=0.2,
                test_fraction=0.2,
                max_num_texts_per_lang=10,
                min_num_chars=10,
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
