'''
Unit test for lang_grad_rev_tag Experiment function in lang_grad_rev_tag module.
'''

import sys
import os
import shutil
import unittest
import tempfile
import argparse
import mufins
from mufins.common.dataset.dataset_file import DatasetFile
from mufins.common.file.csv_file import CsvFile
from mufins.experiments.lang_grad_rev_tag.model import LangGradRevTagModel
from mufins.dataprocs.udpos.preprocess import udpos_preprocess
from mufins.dataprocs.wikipedia.preprocess import wikipedia_preprocess
from mufins.experiments.lang_grad_rev_tag.experiment import (
    LangGradRevTagParameterSpace, lang_grad_rev_tag_experiment,
)
from mufins.common.random.random_number_generator import RandomNumberGenerator


#########################################
get_path = False # pylint: disable=invalid-name
verbose = False # pylint: disable=invalid-name


#########################################
class TestLangGradRevTagExperiment(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the lang_grad_rev_tag_experiment function.
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

            lang_grad_rev_tag_experiment(
                label_src_path=os.path.join(path, 'src_label'),
                lang_src_path=os.path.join(path, 'src_lang'),
                dst_path=os.path.join(path, 'dst'),
                device_name='cpu',
                parameter_space=LangGradRevTagParameterSpace(
                    encoder_name='mock_encoder',
                    layer_index=None,
                    init_stddev=0.1,
                    minibatch_size=2,
                    dropout_rate=0.0,
                    freeze_embeddings=False,
                    encoder_learning_rate=1e-5,
                    postencoder_learning_rate=0.1,
                    grl_lambda=0.0,
                    patience=3,
                    max_epochs=10,

                    attributes_list_or_path=[
                        ('0.4', dict(grl_lambda=0.4)),
                    ],
                ),
                hyperparameter_search_mode=False,
                batch_size=1,
                seed=0,
                verbose=verbose,
                debug_mode=True,
            )
            lang_grad_rev_tag_experiment(
                label_src_path=os.path.join(path, 'src_label'),
                lang_src_path=os.path.join(path, 'src_lang'),
                dst_path=os.path.join(path, 'dst_tune'),
                device_name='cpu',
                parameter_space=LangGradRevTagParameterSpace(
                    encoder_name='mock_encoder',
                    layer_index=None,
                    init_stddev=0.1,
                    minibatch_size=2,
                    dropout_rate=0.0,
                    freeze_embeddings=False,
                    encoder_learning_rate=1e-5,
                    postencoder_learning_rate=0.1,
                    grl_lambda=0.0,
                    patience=3,
                    max_epochs=10,

                    attributes_list_or_path=[
                        ('0.4', dict(grl_lambda=0.4)),
                    ],
                ),
                hyperparameter_search_mode=True,
                batch_size=1,
                seed=0,
                verbose=verbose,
                debug_mode=True,
            )

            if get_path:
                print(path)
                input()
                return

            for exp_id in os.listdir(os.path.join(path, 'dst', 'results')):
                model = LangGradRevTagModel.load_model(
                    model_path=os.path.join(path, 'dst', 'results', exp_id, 'model.json'),
                    device_name='cpu',
                    rng=RandomNumberGenerator(0),
                    params_path=os.path.join(path, 'dst', 'results', exp_id, 'model.pkl'),
                )

                test_set = DatasetFile(
                    os.path.join(path, 'src_label', 'dataset_test.hdf'),
                    model.label_spec,
                )
                test_set.init()
                test_set.load(True)
                (logprobs, _, _) = model.get_label_predictions(test_set, 1)
                test_set.close()

                f = CsvFile(os.path.join(path, 'dst', 'results', exp_id, 'outputs.csv'))
                f.init([
                    'sent_index', 'word_index', 'word',
                    'true_lang', 'true_label', 'pred_label',
                    'logprob_label_ADP', 'logprob_label_AUX', 'logprob_label_DET',
                    'logprob_label_NOUN', 'logprob_label_PART', 'logprob_label_PRON',
                    'logprob_label_PUNC', 'logprob_label_VERB',
                ])
                for (logprobs_row, csv_row) in zip(
                    (token for sent in logprobs for token in sent),
                    f.read(True),
                ):
                    self.assertEqual(['{:.10f}'.format(x) for x in logprobs_row], csv_row[-8:])


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
