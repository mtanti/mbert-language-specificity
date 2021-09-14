'''
Unit test for fine_tune_tag Experiment function in fine_tune_tag module.
'''

import os
import shutil
import unittest
import tempfile
import subprocess
import mufins
from mufins.dataprocs.udpos.preprocess import udpos_preprocess
from mufins.dataprocs.wikipedia.preprocess import wikipedia_preprocess


#########################################
class TestFineTuneTagExperiment(unittest.TestCase):
    '''
    Unit test class.
    '''
    # pylint: disable=no-self-use

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the fine_tune_tag_experiment function.
        '''
        with tempfile.TemporaryDirectory() as path:
            shutil.copytree(
                os.path.join(
                    mufins.path, 'tests', 'dataprocs', 'udpos', 'mock_dataset',
                ),
                os.path.join(path, 'raw_label'),
            )

            shutil.copytree(
                os.path.join(
                    mufins.path, 'tests', 'dataprocs', 'wikipedia', 'mock_dataset',
                ),
                os.path.join(path, 'raw_lang'),
            )

            udpos_preprocess(
                src_path=os.path.join(path, 'raw_label'),
                dst_path=os.path.join(path, 'src_label'),
                tokeniser_name='mock',
                train_fraction=0.6,
                val_fraction=0.4,
                max_num_tokens=7,
                seed=0,
                verbose=False,
            )

            wikipedia_preprocess(
                src_path=os.path.join(path, 'raw_lang'),
                dst_path=os.path.join(path, 'src_lang'),
                tokeniser_name='mock',
                train_fraction=0.4,
                val_fraction=0.2,
                dev_fraction=0.2,
                test_fraction=0.2,
                max_num_texts_per_lang=10,
                min_num_chars=10,
                max_num_tokens=7,
                seed=0,
                verbose=False,
            )

            with open(os.path.join(path, 'param_space.txt'), 'w', encoding='utf-8') as f:
                print('''\
exp\t{}
''', file=f)

            subprocess.run([
                'python',
                os.path.join(
                    mufins.path, '..', 'bin', 'experiments', 'fine_tune_tag', 'experiment.py'
                ),
                '--label_src_path', os.path.join(path, 'src_label'),
                '--lang_src_path', os.path.join(path, 'src_lang'),
                '--dst_path', os.path.join(path, 'dst'),
                '--device_name', 'cpu',
                '--default_encoder_name', 'mock_encoder',
                '--default_init_stddev', '0.1',
                '--default_minibatch_size', '2',
                '--default_dropout_rate', '0.0',
                '--default_freeze_embeddings', 'no',
                '--default_encoder_learning_rate', '1e-5',
                '--default_postencoder_learning_rate', '0.1',
                '--default_patience', '3',
                '--default_max_epochs', '10',
                '--parameter_space_path', os.path.join(path, 'param_space.txt'),
                '--hyperparameter_search_mode', 'no',
                '--batch_size', '1',
                '--seed', '0',
                '--verbose', 'no',
                '--debug_mode', 'yes',
            ], check=True)

            subprocess.run([
                'python',
                os.path.join(
                    mufins.path, '..', 'bin', 'experiments', 'fine_tune_tag', 'experiment.py'
                ),
                '--label_src_path', os.path.join(path, 'src_label'),
                '--lang_src_path', os.path.join(path, 'src_lang'),
                '--dst_path', os.path.join(path, 'dst_tune'),
                '--device_name', 'cpu',
                '--default_encoder_name', 'mock_encoder',
                '--default_init_stddev', '0.1',
                '--default_minibatch_size', '2',
                '--default_dropout_rate', '0.0',
                '--default_freeze_embeddings', 'no',
                '--default_encoder_learning_rate', '1e-5',
                '--default_postencoder_learning_rate', '0.1',
                '--default_patience', '3',
                '--default_max_epochs', '10',
                '--parameter_space_path', os.path.join(path, 'param_space.txt'),
                '--hyperparameter_search_mode', 'yes',
                '--batch_size', '1',
                '--seed', '0',
                '--verbose', 'no',
                '--debug_mode', 'yes',
            ], check=True)


#########################################
if __name__ == '__main__':
    unittest.main()
