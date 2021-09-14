'''
Unit test for Preprocess XNLI Dataset function in XNLI module.
'''

import shutil
import unittest
import os
import tempfile
import subprocess
import mufins


#########################################
class TestPreprocessXNLIDataset(unittest.TestCase):
    '''
    Unit test class.
    '''
    # pylint: disable=no-self-use

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the preprocess_xnli_dataset function.
        '''
        with tempfile.TemporaryDirectory() as path:
            shutil.copytree(
                os.path.join(
                    mufins.path, 'tests', 'dataprocs', 'xnli', 'mock_dataset',
                ),
                os.path.join(path, 'src'),
            )

            subprocess.run([
                'python',
                os.path.join(
                    mufins.path, '..', 'bin', 'dataprocs', 'xnli', 'preprocess.py'
                ),
                '--src_path_xnli', os.path.join(path, 'src', 'XNLI-1.0'),
                '--src_path_multinli', os.path.join(path, 'src', 'multinli_1.0'),
                '--dst_path', os.path.join(path, 'dst'),
                '--tokeniser_name', 'mock',
                '--train_fraction', '0.6',
                '--val_fraction', '0.4',
                '--max_num_tokens', '14',
                '--seed', '0',
                '--verbose', 'no',
                '--debug_mode', 'yes',
            ], check=True)


#########################################
if __name__ == '__main__':
    unittest.main()
