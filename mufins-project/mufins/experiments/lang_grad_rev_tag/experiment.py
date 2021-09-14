'''
Run experiment with ID: lang_grad_rev_tag
'''

import os
import shutil
from typing import List, Optional, cast
import torch
from mufins.common.random.random_number_generator import RandomNumberGenerator
from mufins.common.time.time_utils import Timer, get_readable_duration
from mufins.common.dataset.dataset import Dataset, autocloser
from mufins.common.dataset.dataset_file import DatasetFile
from mufins.common.checkpoint.checkpoint_manager import CheckpointManager
from mufins.common.log.log import Log
from mufins.common.log.log_file import LogFile
from mufins.common.log.log_cli import LogCli
from mufins.common.log.log_composite import LogComposite
from mufins.common.file.csv_file import CsvFile
from mufins.common.file.about_file import AboutFile
from mufins.common.file.version_file import VersionFile
from mufins.dataprocs.udpos.preprocess import DATASET_VERSION as UDPOS_DATASET_VERSION
from mufins.dataprocs.wikipedia.preprocess import DATASET_VERSION as WILI_DATASET_VERSION
from mufins.dataprocs.udpos.data_row import UDPOSDataRow
from mufins.dataprocs.wikipedia.data_row import WikipediaDataRow
from mufins.dataprocs.udpos.data_spec import UDPOSDataSpec
from mufins.dataprocs.wikipedia.data_spec import WikipediaDataSpec
from mufins.experiments.lang_grad_rev_tag.hyperparameters import LangGradRevTagParameterSpace
from mufins.experiments.lang_grad_rev_tag.model import LangGradRevTagModel
from mufins.experiments.lang_grad_rev_tag.evaluate import (
    prepare_encoding_plot, encoding_plot_label, encoding_plot_lang, eval_label, eval_lang
)


#########################################
def _output_results(
    dst_path: str,
    exp_id: str,
    label_spec: UDPOSDataSpec,
    lang_spec: WikipediaDataSpec,
    dset_label_test: Dataset[UDPOSDataRow],
    dset_lang_test: Dataset[WikipediaDataRow],
    model: LangGradRevTagModel,
    hyperparameter_search_mode: bool,
    batch_size: int,
    results_file: CsvFile,
    log: Log,
) -> None:
    '''
    Calculate the evaluation results of a single model and save outputs to file.

    :param dst_path: The path to store the results.
    :param exp_id: A unique name for the experiment.
    :param label_spec: The data specifier for labels.
    :param lang_spec: The data specifier for languages.
    :param dset_label_test: The labels test set to test on.
    :param dset_lang_test: The languages test set to test on.
    :param model: The trained model to evaluate.
    :param hyperparameter_search_mode: Whether to enter into hyperparameter search mode where
        minimal output and evaluation is produced.
    :param batch_size: The maximum number of sentences to process at once.
    :param results_file: The CSV file to store results in.
    :param log: The log to send progress information to.
    '''
    if not hyperparameter_search_mode:
        prepare_encoding_plot(
            path=os.path.join(dst_path, 'results', exp_id, 'plots', 'test'),
            label_spec=label_spec,
            lang_spec=lang_spec,
        )
        (label_label_vmeasure, label_lang_vmeasure) = encoding_plot_label(
            path=os.path.join(dst_path, 'results', exp_id, 'plots', 'test'),
            label_spec=label_spec,
            lang_spec=lang_spec,
            dset_label=dset_label_test,
            label_encoder=model.get_label_encodings,
            batch_size=batch_size,
            log=log,
        )
        lang_vmeasure = encoding_plot_lang(
            path=os.path.join(dst_path, 'results', exp_id, 'plots', 'test'),
            lang_spec=lang_spec,
            dset_lang=dset_lang_test,
            lang_encoder=model.get_lang_encodings,
            batch_size=batch_size,
            log=log,
        )

    (
        label_label_score_f1_macro,
        label_label_score_f1_micro,
        label_label_score_f1_macro_by_lang,
        label_lang_score_f1_macro,
    ) = eval_label(
        label_spec=label_spec,
        lang_spec=lang_spec,
        dset_label=dset_label_test,
        label_predictor=model.get_label_predictions,
        lang_predictor=model.get_lang_predictions,
        hyperparameter_search_mode=hyperparameter_search_mode,
        batch_size=batch_size,
        log=log,
        path=os.path.join(dst_path, 'results', exp_id),
    )

    if not hyperparameter_search_mode:
        (
            lang_score_f1_macro, _, _, _,
        ) = eval_lang(
            lang_spec=lang_spec,
            dset_lang=dset_lang_test,
            lang_predictor=model.get_lang_predictions,
            batch_size=batch_size,
            log=log,
            path=os.path.join(dst_path, 'results', exp_id),
        )

    if not hyperparameter_search_mode:
        results_file.append(
            [
                exp_id,
                '{:.10f}'.format(label_label_score_f1_macro),
                '{:.10f}'.format(label_label_score_f1_micro),
                '{:.10f}'.format(label_lang_score_f1_macro),
                '{:.10f}'.format(lang_score_f1_macro),
                '{:.10f}'.format(label_label_vmeasure),
                '{:.10f}'.format(label_lang_vmeasure),
                '{:.10f}'.format(lang_vmeasure),
            ] + [
                (
                    '{:.10f}'.format(cast(float, label_label_score_f1_macro_by_lang[lang]))
                    if label_label_score_f1_macro_by_lang[lang] is not None
                    else ''
                )
                for lang in label_spec.lang_names
            ]
        )
    else:
        results_file.append(
            [
                exp_id,
                '{:.10f}'.format(label_label_score_f1_macro),
            ]
        )


#########################################
def lang_grad_rev_tag_experiment(
    label_src_path: str,
    lang_src_path: str,
    dst_path: str,
    device_name: str,
    parameter_space: LangGradRevTagParameterSpace,
    hyperparameter_search_mode: bool,
    batch_size: int,
    seed: Optional[int],
    verbose: bool = True,
    debug_mode: bool = True,
) -> None:
    '''
    Run experiment lang_grad_rev_tag.

    :param label_src_path: The path to the directory with the preprocessed UDPOS dataset.
    :param lang_src_path: The path to the directory with the preprocessed Wikipedia dataset.
    :param dst_path: The path to the directory to contain the experiment
        results.
    :param device_name: The GPU device name to use such as 'cuda:0'. Use 'cpu'
        if no GPU is available.
    :param parameter_space: The parameter space to traverse in the experiment.
    :param hyperparameter_search_mode: Whether to enter into hyperparameter search mode where the
        dev set is used instead of the test set and minimal output and evaluation is produced.
    :param batch_size: The maximum amount of sentences to process in one go.
    :param seed: The seed to use for the random number generator.
        If None then a randomly generated seed will be used.
    :param verbose: Whether to show console output.
    :param debug_mode: Whether to show full error information.
    '''
    log = None
    try:
        os.makedirs(dst_path, exist_ok=True)

        log = LogComposite(
            cast(List[Log], [])
            + [LogFile(os.path.join(dst_path, 'log.txt'), show_progress=False)]
            + ([LogCli()] if verbose else [])
        )
        log.init()

        with Timer() as overall_timer:
            log.log_message('')
            log.log_message('------------------------------------------------')
            log.log_message('Running lang_grad_rev_tag experiment.')
            log.log_message('')

            checkpoint_manager = CheckpointManager(
                os.path.join(dst_path, 'checkpoint.sqlite3')
            )
            checkpoint_manager.init()

            AboutFile(os.path.join(dst_path, 'about.csv')).init()

            with open(os.path.join(label_src_path, 'data_spec.json'), 'r', encoding='utf-8') as f:
                label_spec = UDPOSDataSpec.from_json(f.read().strip())
            with open(os.path.join(lang_src_path, 'data_spec.json'), 'r', encoding='utf-8') as f:
                lang_spec = WikipediaDataSpec.from_json(f.read().strip())
            if not set(label_spec.lang_names) <= set(lang_spec.lang_names):
                raise ValueError(
                    'Label data set languages must be a subset of the language data set languages.'
                )

            results_file = CsvFile(os.path.join(dst_path, 'results.csv'))
            if not hyperparameter_search_mode:
                results_file.init([
                    'exp_id',
                    'label_label_macro_f1_score',
                    'label_label_micro_f1_score',
                    'label_lang_macro_f1_score',
                    'lang_macro_f1_score',
                    'label_label_vmeasure',
                    'label_lang_vmeasure',
                    'lang_vmeasure',
                ] + [
                    'label_macro_f1_score_{}'.format(lang)
                    for lang in label_spec.lang_names
                ])
            else:
                results_file.init([
                    'exp_id',
                    'label_label_macro_f1_score',
                ])

            if seed is None:
                seed = RandomNumberGenerator.make_seed()

            log.log_message('Seed: {}'.format(seed))
            log.log_message('Label source path: {}'.format(label_src_path))
            log.log_message('Language source path: {}'.format(lang_src_path))
            log.log_message('Destination path: {}'.format(dst_path))
            log.log_message('Device name: {}'.format(device_name))
            log.log_message('Hyperparameter search mode: {}'.format(hyperparameter_search_mode))
            log.log_message('Batch size: {}'.format(batch_size))
            log.log_message('GPU available?: {}'.format(torch.cuda.is_available()))
            log.log_message('')

            log.log_message('Default parameters:')
            for (key, value) in parameter_space.default_attributes.items():
                log.log_message('> {}: {}'.format(key, value))
            log.log_message('')

            log.log_message('Loading data.')

            version_file = VersionFile(os.path.join(label_src_path, 'dataset_version.txt'))
            version_file.init()
            if UDPOS_DATASET_VERSION != version_file.read():
                raise ValueError('Label dataset is not of the expected version.')

            version_file = VersionFile(os.path.join(lang_src_path, 'dataset_version.txt'))
            version_file.init()
            if WILI_DATASET_VERSION != version_file.read():
                raise ValueError('Language dataset is not of the expected version.')

            dset_label_train = DatasetFile[UDPOSDataRow](
                os.path.join(label_src_path, 'dataset_train.hdf'),
                label_spec,
            )
            dset_label_val = DatasetFile[UDPOSDataRow](
                os.path.join(label_src_path, 'dataset_val.hdf'),
                label_spec,
            )
            if not hyperparameter_search_mode:
                dset_label_test = DatasetFile[UDPOSDataRow](
                    os.path.join(label_src_path, 'dataset_test.hdf'),
                    label_spec,
                )
            else:
                dset_label_test = DatasetFile[UDPOSDataRow](
                    os.path.join(label_src_path, 'dataset_dev.hdf'),
                    label_spec,
                )

            dset_lang_train = DatasetFile[WikipediaDataRow](
                os.path.join(lang_src_path, 'dataset_train.hdf'),
                lang_spec,
            )
            dset_lang_val = DatasetFile[WikipediaDataRow](
                os.path.join(lang_src_path, 'dataset_val.hdf'),
                lang_spec,
            )
            if not hyperparameter_search_mode:
                dset_lang_test = DatasetFile[WikipediaDataRow](
                    os.path.join(lang_src_path, 'dataset_test.hdf'),
                    lang_spec,
                )
            else:
                dset_lang_test = DatasetFile[WikipediaDataRow](
                    os.path.join(lang_src_path, 'dataset_dev.hdf'),
                    lang_spec,
                )

            with autocloser([
                dset_label_train,
                dset_label_val,
                dset_label_test,
            ]):
                with autocloser([
                    dset_lang_train,
                    dset_lang_val,
                    dset_lang_test,
                ]):
                    dset_label_train.init()
                    dset_label_val.init()
                    dset_label_test.init()
                    dset_lang_train.init()
                    dset_lang_val.init()
                    dset_lang_test.init()

                    dset_label_train.load(as_readonly=True)
                    dset_label_val.load(as_readonly=True)
                    dset_label_test.load(as_readonly=True)
                    dset_lang_train.load(as_readonly=True)
                    dset_lang_val.load(as_readonly=True)
                    dset_lang_test.load(as_readonly=True)

                    log.log_message('Starting experiments.')
                    log.log_message('')

                    for (exp_id, hyperparams) in parameter_space:
                        log.log_message('----------------')
                        log.log_message('Experiment: {}'.format(exp_id))
                        log.log_message('')

                        rng = RandomNumberGenerator(seed)

                        with checkpoint_manager.checkpoint(exp_id) as handle:
                            if handle.was_found_ready():
                                log.log_message('Was found ready.')
                                log.log_message('')
                                handle.skip()

                            os.makedirs(os.path.join(dst_path, 'results', exp_id), exist_ok=True)

                            with Timer() as exp_timer:
                                log.log_message('Creating model.')
                                model = LangGradRevTagModel(
                                    label_spec=label_spec,
                                    lang_spec=lang_spec,
                                    device_name=device_name,
                                    hyperparams=hyperparams,
                                    rng=rng,
                                )
                                model.save_model(
                                    os.path.join(dst_path, 'results', exp_id, 'model.json')
                                )
                                log.log_message('')

                                log.log_message('Training model.')
                                with Timer() as training_timer:
                                    model.fit(
                                        checkpoint_id='{}/train'.format(exp_id),
                                        dset_label_train=dset_label_train,
                                        dset_lang_train=dset_lang_train,
                                        dset_label_val=dset_label_val,
                                        dset_lang_val=dset_lang_val,
                                        batch_size=batch_size,
                                        hyperparameter_search_mode=hyperparameter_search_mode,
                                        model_path=os.path.join(
                                            dst_path, 'results', exp_id
                                        ),
                                        train_history_path=os.path.join(
                                            dst_path, 'results', exp_id
                                        ),
                                        checkpoint_manager=checkpoint_manager,
                                        log=log,
                                    )
                                log.log_message('Duration: {}'.format(
                                    get_readable_duration(training_timer.get_duration())
                                ))
                                log.log_message('')

                                log.log_message('Evaluating model.')
                                with Timer() as eval_timer:
                                    _output_results(
                                        dst_path=dst_path,
                                        exp_id=exp_id,
                                        label_spec=label_spec,
                                        lang_spec=lang_spec,
                                        dset_label_test=dset_label_test,
                                        dset_lang_test=dset_lang_test,
                                        model=model,
                                        hyperparameter_search_mode=hyperparameter_search_mode,
                                        batch_size=batch_size,
                                        results_file=results_file,
                                        log=log,
                                    )
                                    log.log_message('Duration: {}'.format(
                                    get_readable_duration(eval_timer.get_duration())
                                ))
                            log.log_message('')
                            log.log_message('Duration: {}'.format(
                                get_readable_duration(exp_timer.get_duration())
                            ))
                            log.log_message('')

                            if hyperparameter_search_mode:
                                shutil.rmtree(os.path.join(dst_path, 'results', exp_id))

                    if hyperparameter_search_mode:
                        shutil.rmtree(os.path.join(dst_path, 'results'))

        log.log_message('')
        log.log_message('Ready.')
        log.log_message('Duration: {}'.format(
            get_readable_duration(overall_timer.get_duration())
        ))
    except Exception as ex:
        if log is not None:
            log.log_error('ERROR: {}'.format(ex))
        if debug_mode or log is None:
            raise
