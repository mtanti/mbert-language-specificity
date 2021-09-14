'''
Main data preprocessor for the XNLI dataset.
'''

import os
import json
from typing import List, Sequence, Tuple, Mapping, cast, Optional
from mufins.common.random.random_number_generator import RandomNumberGenerator
from mufins.common.dataset.dataset import Dataset, autocloser
from mufins.common.dataset.dataset_file import DatasetFile
from mufins.common.log.log import Log
from mufins.common.log.log_file import LogFile
from mufins.common.log.log_cli import LogCli
from mufins.common.log.log_composite import LogComposite
from mufins.common.time.time_utils import Timer, get_readable_duration
from mufins.common.file.csv_file import CsvFile
from mufins.common.file.about_file import AboutFile
from mufins.common.file.version_file import VersionFile
from mufins.common.tokeniser.tokeniser import Tokeniser
from mufins.dataprocs.standard_langs import ISO_639_1 as STD_LANGS
from mufins.dataprocs.xnli.data_spec import XNLIDataSpec
from mufins.dataprocs.xnli.data_row import XNLIDataRow


#########################################
DATASET_VERSION = 5


#########################################
def _extract_language_set(
    src_path_xnli: str,
) -> Sequence[str]:
    '''
    Get all the different languages used in the dataset.

    :param src_path_xnli: The path to the raw XNLI dataset.
    :return: A list of unique language names.
    '''
    language_set = set()
    with open(os.path.join(src_path_xnli, 'xnli.test.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            fields = json.loads(line)
            language_set.add(fields['language'])
    return sorted(language_set & STD_LANGS)


#########################################
def _extract_label_set(
    src_path_xnli: str,
) -> Sequence[str]:
    '''
    Get all the different labels used in the dataset.

    :param src_path_xnli: The path to the raw XNLI dataset.
    :return: A list of unique label names.
    '''
    label_set = set()
    with open(os.path.join(src_path_xnli, 'xnli.test.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            fields = json.loads(line)
            if fields['language'] not in STD_LANGS:
                continue
            label_set.add(fields['gold_label'])
    return sorted(label_set)


#########################################
def _analyse_data(
    src_path_xnli: str,
    src_path_multinli: str,
    tokeniser: Tokeniser,
) -> Tuple[Mapping[str, int], int]:
    '''
    Get the total number of rows and the maximum sentence length.

    :param src_path_xnli: The path to the raw XNLI dataset.
    :param src_path_multinli: The path to the raw multi NLI dataset.
    :param tokeniser: The tokeniser to use.
    :return: A rows-len pair where rows is the total number of rows and len is
        the maximum sentence length.
    '''
    num_rows = {'train': 0, 'dev': 0, 'test': 0}
    max_len = 0
    for (split, fname) in [
        ('train', os.path.join(src_path_multinli, 'multinli_1.0_train.jsonl')),
        ('dev', os.path.join(src_path_xnli, 'xnli.dev.jsonl')),
        ('test', os.path.join(src_path_xnli, 'xnli.test.jsonl')),
    ]:
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue

                fields = json.loads(line)

                lang_name = fields['language'] if 'language' in fields else 'en'
                if lang_name not in STD_LANGS:
                    continue
                if not tokeniser.all_known_tokens(fields['sentence1']):
                    continue
                if not tokeniser.all_known_tokens(fields['sentence2']):
                    continue

                text_len = len(
                    tokeniser.indexify_text_pair(fields['sentence1'], fields['sentence2'])
                )

                num_rows[split] += 1
                max_len = max(max_len, text_len)

    return (num_rows, max_len)


#########################################
def _transfer_data(
    src_path_xnli: str,
    src_path_multinli: str,
    dst_path: str,
    spec: XNLIDataSpec,
    num_rows: Mapping[str, int],
    log: Log,
) -> None:
    '''
    Transfer all the raw data into the dataset.

    :param src_path_xnli: The path to the raw XNLI dataset.
    :param src_path_multinli: The path to the raw multi NLI dataset.
    :param dst_path: The path to store the processed dataset.
    :param spec: The data specification.
    :param num_rows: A mapping from data split to number of rows in the split.
    :param log: The log to pass progress information to.
    '''
    dset = {
        'test': DatasetFile[XNLIDataRow](
            os.path.join(dst_path, 'dataset_test.hdf'), spec
        ),
        'dev': DatasetFile[XNLIDataRow](
            os.path.join(dst_path, 'dataset_dev.hdf'), spec
        ),
        'train': DatasetFile[XNLIDataRow](
            os.path.join(dst_path, 'dataset_train-orig.hdf'), spec
        )
    }

    with autocloser(list(dset.values())):
        log.log_message('> Creating test dataset.')
        dset['test'].init(num_rows['test'])

        log.log_message('> Creating dev dataset.')
        dset['dev'].init(num_rows['dev'])

        log.log_message('> Creating train-orig dataset.')
        dset['train'].init(num_rows['train'])

        dset['test'].load(as_readonly=False)
        dset['dev'].load(as_readonly=False)
        dset['train'].load(as_readonly=False)

        for (dset_split, dset_fname, src_fname) in [
            ('train', 'train-orig', os.path.join(src_path_multinli, 'multinli_1.0_train.jsonl')),
            ('dev', 'dev', os.path.join(src_path_xnli, 'xnli.dev.jsonl')),
            ('test', 'test', os.path.join(src_path_xnli, 'xnli.test.jsonl')),
        ]:
            log.log_message('> Transferring data to {} dataset.'.format(dset_fname))
            dset_obj = dset[dset_split]
            row_num = 0
            log.progress_start(0, num_rows[dset_split])
            with open(src_fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue

                    fields = json.loads(line)
                    premise_text = fields['sentence1']
                    hypothesis_text = fields['sentence2']
                    label = fields['gold_label']
                    lang_name = fields['language'] if 'language' in fields else 'en'

                    if lang_name not in STD_LANGS:
                        continue
                    if not spec.tokeniser.all_known_tokens(fields['sentence1']):
                        continue
                    if not spec.tokeniser.all_known_tokens(fields['sentence2']):
                        continue

                    row = XNLIDataRow(
                        lang_name,
                        premise_text,
                        hypothesis_text,
                        label,
                    )
                    preprocessed = dset_obj.spec.preprocess(row)
                    dset_obj.set_row(row_num, preprocessed)
                    row_num += 1

                    log.progress_update(row_num)

            log.progress_end()


#########################################
def _split_data(
    dst_path: str,
    spec: XNLIDataSpec,
    train_fraction: float,
    val_fraction: float,
    rng: RandomNumberGenerator,
    log: Log,
) -> None:
    '''
    Take the monolitic dataset and split it into train-val-dev-test splits.

    :param dst_path: The path to store the processed dataset.
    :param spec: The data specification associated with the dataset.
    :param train_fraction: The fraction of the original training set to use for the training set.
    :param val_fraction: The fraction of the original training set to use for the validation set.
    :param rng: The random number generator to use.
    :param log: The log to receive progress information.
    '''
    dset_train = DatasetFile[XNLIDataRow](
        os.path.join(dst_path, 'dataset_train-orig.hdf'), spec
    )
    with autocloser([dset_train]):
        dset_train.init()
        dset_train.load(as_readonly=True)

        def factory(
            name: str,
            size: int,
        ) -> Dataset[XNLIDataRow]:
            sub_dataset = DatasetFile[XNLIDataRow](
                os.path.join(dst_path, 'dataset_{}.hdf'.format(name)),
                spec,
            )
            sub_dataset.init(size)
            sub_dataset.load(as_readonly=False)
            return sub_dataset

        log.log_message('> Creating base split.')
        dset_train.split(
            partition_fractions=[
                ('train', train_fraction),
                ('val', val_fraction),
            ],
            dataset_factory=factory,
            stratification_key=lambda row:row['lang'].tolist(),
            rng=rng,
            log=log,
        )


#########################################
def _output_readable_dataset(
    dst_path: str,
    spec: XNLIDataSpec,
    log: Log,
) -> None:
    '''
    Output a human readable version of the contents of the dataset.

    :param dst_path: The path to the processed dataset folder.
    :param spec: The data specification associated with the dataset.
    :param log: The log to pass progress information to.
    '''
    dset_train_orig = DatasetFile[XNLIDataRow](
        os.path.join(dst_path, 'dataset_train-orig.hdf'), spec
    )
    dset_train = DatasetFile[XNLIDataRow](
        os.path.join(dst_path, 'dataset_train.hdf'), spec
    )
    dset_val = DatasetFile[XNLIDataRow](
        os.path.join(dst_path, 'dataset_val.hdf'), spec
    )
    dset_dev = DatasetFile[XNLIDataRow](
        os.path.join(dst_path, 'dataset_dev.hdf'), spec
    )
    dset_test = DatasetFile[XNLIDataRow](
        os.path.join(dst_path, 'dataset_test.hdf'), spec
    )

    with autocloser([
        dset_train_orig,
        dset_train,
        dset_val,
        dset_dev,
        dset_test,
    ]):
        dset_train_orig.init()
        dset_train.init()
        dset_val.init()
        dset_dev.init()
        dset_test.init()

        dset_train_orig.load(as_readonly=True)
        dset_train.load(as_readonly=True)
        dset_val.load(as_readonly=True)
        dset_dev.load(as_readonly=True)
        dset_test.load(as_readonly=True)

        for (dset_name, dset_obj) in [
            ('train-orig', dset_train_orig),
            ('train', dset_train),
            ('val', dset_val),
            ('dev', dset_dev),
            ('test', dset_test),
        ]:
            log.log_message('')
            log.log_message('Dataset: {}'.format(dset_name))

            if dset_name != 'train-orig':
                data_file = CsvFile(os.path.join(dst_path, 'dataset_{}.csv'.format(dset_name)))
                data_file.init(
                    ['row', 'language', 'tokens', 'label']
                )

            lang_freqs = {lang_name: 0 for lang_name in spec.lang_names}
            label_freqs = {label_name: 0 for label_name in spec.label_names}
            log.progress_start(0, dset_obj.size)
            for i in range(dset_obj.size):
                row = dset_obj.get_row(i)
                row_lang_name = spec.lang_names[row['lang'].tolist()]
                row_label_name = spec.label_names[row['label'].tolist()]
                if dset_name != 'train-orig':
                    data_file.append([
                        i,
                        row_lang_name,
                        ' '.join(
                            spec.tokeniser.textify_indexes(
                                row['tokens'][:row['num_tokens']][spec.tokeniser.content_slice]
                            )[0]
                        ),
                        row_label_name,
                    ])
                lang_freqs[row_lang_name] += 1
                label_freqs[row_label_name] += 1

                log.progress_update(i)

            log.progress_end()

            log.log_message('Number of rows: {}'.format(dset_obj.size))
            log.log_message('Language frequencies:')
            for lang_name in spec.lang_names:
                log.log_message('> {}: {}'.format(
                    lang_name, lang_freqs[lang_name]
                ))
            log.log_message('Label frequencies:')
            for label_name in spec.label_names:
                log.log_message('> {}: {}'.format(
                    label_name,
                    label_freqs[label_name],
                ))


#########################################
def xnli_preprocess(
    src_path_xnli: str,
    src_path_multinli: str,
    dst_path: str,
    tokeniser_name: str,
    train_fraction: float,
    val_fraction: float,
    max_num_tokens: int,
    seed: Optional[int],
    verbose: bool = True,
    debug_mode: bool = True,
) -> None:
    '''
    Save the entire XNLI dataset of files in preprocessed form.

    :param src_path_xnli: The path to the raw XNLI dataset.
    :param src_path_multinli: The path to the raw multi NLI dataset.
    :param dst_path: The path to the directory to contain the preprocessed dataset.
    :param tokeniser_name: The tokeniser to use.
    :param train_fraction: The fraction of the original training set to use for the training set.
    :param val_fraction: The fraction of the original training set to use for the validation set.
    :param max_num_tokens: The maximum number of tokens in a sentence beyond which is trimmed.
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
            log.log_message('Preprocessing XNLI dataset.')
            log.log_message('')

            AboutFile(os.path.join(dst_path, 'about.csv')).init()
            VersionFile(os.path.join(dst_path, 'dataset_version.txt')).init(DATASET_VERSION)
            if seed is None:
                seed = RandomNumberGenerator.make_seed()

            log.log_message('Dataset version: {}'.format(DATASET_VERSION))
            log.log_message('Seed: {}'.format(seed))
            log.log_message('Source path XNLI: {}'.format(src_path_xnli))
            log.log_message('Source path multi-NLI: {}'.format(src_path_multinli))
            log.log_message('Destination path: {}'.format(dst_path))
            log.log_message('Tokeniser: {}'.format(tokeniser_name))
            log.log_message('Splits')
            log.log_message('> Train: {:.2%}'.format(train_fraction))
            log.log_message('> Val: {:.2%}'.format(val_fraction))
            log.log_message('Maximum number of tokens: {}'.format(max_num_tokens))
            log.log_message('')

            rng = RandomNumberGenerator(seed)

            log.log_message('Extracting label set.')
            label_names = _extract_label_set(src_path_xnli)

            log.log_message('Extracting language set.')
            lang_names = _extract_language_set(src_path_xnli)

            log.log_message('Creating data specification.')
            spec = XNLIDataSpec(
                lang_names, max_num_tokens, tokeniser_name, label_names
            )

            with open(os.path.join(dst_path, 'lang_names.txt'), 'w', encoding='utf-8') as f_lang:
                print('\n'.join(spec.lang_names), file=f_lang)
            with open(os.path.join(dst_path, 'label_names.txt'), 'w', encoding='utf-8') as f_label:
                print('\n'.join(spec.label_names), file=f_label)
            with open(os.path.join(dst_path, 'data_spec.json'), 'w', encoding='utf-8') as f_spec:
                print(spec.to_json(), file=f_spec)

            log.log_message('Analysing data.')
            (num_rows, max_len) = _analyse_data(src_path_xnli, src_path_multinli, spec.tokeniser)
            log.log_message('> Longest sentence: {}'.format(max_len))
            log.log_message('')

            log.log_message('Transferring data.')
            _transfer_data(src_path_xnli, src_path_multinli, dst_path, spec, num_rows, log)
            log.log_message('')

            log.log_message('Splitting dataset.')
            _split_data(dst_path, spec, train_fraction, val_fraction, rng, log)
            log.log_message('')

            log.log_message('Outputting human readable dataset and label counts.')
            _output_readable_dataset(dst_path, spec, log)
            log.log_message('')

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
