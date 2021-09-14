'''
Unit test for Preprocess Wikipedia Dataset function in wikipedia module.
'''

import shutil
import unittest
import os
import tempfile
import subprocess
import mufins


#########################################
class TestPreprocessWikipediaDataset(unittest.TestCase):
    '''
    Unit test class.
    '''
    # pylint: disable=no-self-use

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the preprocess_wikipedia_dataset function.
        '''
        with tempfile.TemporaryDirectory() as path:
            shutil.copytree(
                os.path.join(
                    mufins.path, 'tests', 'dataprocs', 'wikipedia', 'mock_dataset',
                ),
                os.path.join(path, 'src'),
            )

            subprocess.run([
                'python',
                os.path.join(
                    mufins.path, '..', 'bin', 'dataprocs', 'wikipedia', 'preprocess.py'
                ),
                '--src_path', os.path.join(path, 'src'),
                '--dst_path', os.path.join(path, 'dst'),
                '--tokeniser_name', 'mock',
                '--train_fraction', '0.4',
                '--val_fraction', '0.2',
                '--dev_fraction', '0.2',
                '--test_fraction', '0.2',
                '--max_num_texts_per_lang', '10',
                '--min_num_chars', '10',
                '--max_num_tokens', '7',
                '--seed', '0',
                '--verbose', 'no',
                '--debug_mode', 'yes',
            ], check=True)


#########################################
if __name__ == '__main__':
    unittest.main()
