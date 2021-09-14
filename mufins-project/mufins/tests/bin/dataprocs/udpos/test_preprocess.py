'''
Unit test for Preprocess UDPOS Dataset function in udpos module.
'''

import shutil
import unittest
import os
import tempfile
import subprocess
import mufins


#########################################
class TestPreprocessUDPOSDataset(unittest.TestCase):
    '''
    Unit test class.
    '''
    # pylint: disable=no-self-use

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the preprocess_udpos_dataset function.
        '''
        with tempfile.TemporaryDirectory() as path:
            shutil.copytree(
                os.path.join(
                    mufins.path, 'tests', 'dataprocs', 'udpos', 'mock_dataset',
                ),
                os.path.join(path, 'src'),
            )

            subprocess.run([
                'python',
                os.path.join(
                    mufins.path, '..', 'bin', 'dataprocs', 'udpos', 'preprocess.py'
                ),
                '--src_path', os.path.join(path, 'src'),
                '--dst_path', os.path.join(path, 'dst'),
                '--tokeniser_name', 'mock',
                '--train_fraction', '0.6',
                '--val_fraction', '0.4',
                '--max_num_tokens', '7',
                '--seed', '0',
                '--verbose', 'no',
                '--debug_mode', 'yes',
            ], check=True)


#########################################
if __name__ == '__main__':
    unittest.main()
