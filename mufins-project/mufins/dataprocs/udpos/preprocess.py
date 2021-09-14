'''
Main data preprocessor for the UDPOS dataset.
'''

import os
import collections
import glob
import re
from typing import List, Sequence, Tuple, Counter, Mapping, cast, Optional
import conllu
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
from mufins.dataprocs.udpos.data_spec import UDPOSDataSpec
from mufins.dataprocs.udpos.data_row import UDPOSDataRow


#########################################
DATASET_VERSION = 2


#########################################
def _get_files(
    path: str,
) -> Mapping[str, Sequence[Tuple[str, str]]]:
    '''
    Get the files and languages of the files from the raw dataset.

    :param path: The path to the raw dataset.
    :return: A dictionary mapping dataset splits to filename-language pairs.
    '''
    result = collections.defaultdict(list)
    for fpath in glob.iglob(os.path.join(path, 'ud-treebanks-v2.7', '*', '*.conllu')):
        fname = os.path.basename(fpath)
        match = re.match(r'([a-z]+)_.*-([a-z]+)\.conllu', fname)
        assert match is not None
        (lang, split) = match.groups()
        if lang in STD_LANGS:
            result[split].append((fpath, lang))
    for files in result.values():
        files.sort()
    return result


#########################################
def _parse_conllu(
    path: str,
) -> Tuple[
    Sequence[Sequence[str]],
    Sequence[Sequence[str]],
]:
    '''
    Get the relevant data from a CONLLU text file.

    :param path: The path to the CONLLU file.
    :return: A pair consisting of a sequence of sentences (list of words) and a sequence of sentence
        tags (list of part of speech tags).
    '''
    with open(os.path.join(path, path), 'r', encoding='utf-8') as f:
        data = conllu.parse(f.read())
    clean_data = [
        [
            (token['form'], token['upos'])
            for token in sent
            if isinstance(token['id'], int)
        ]
        for sent in data
        if (
            not all(token['form'] == '_' for token in sent if isinstance(token['id'], int))
            or any(token['upos'] == '_' for token in sent if isinstance(token['id'], int))
        )
    ]
    return (
        [[word for (word, tag) in sent] for sent in clean_data],
        [[tag for (word, tag) in sent] for sent in clean_data],
    )


#########################################
def _extract_language_set(
    path: str,
) -> Sequence[str]:
    '''
    Get all the different languages used in the dataset.

    :param path: The path to the raw dataset.
    :return: A list of unique language names.
    '''
    language_set = set()
    for (_, files) in _get_files(path).items():
        for (_, lang) in files:
            language_set.add(lang)
    return sorted(language_set)


#########################################
def _extract_label_set(
    path: str,
) -> Sequence[str]:
    '''
    Get all the different labels used in the dataset.

    :param path: The path to the raw dataset.
    :return: A list of unique label names.
    '''
    label_set = set()
    for (split, files) in _get_files(path).items():
        if split != 'train':
            continue
        for (fname, _) in files:
            for sent_tags in _parse_conllu(fname)[1]:
                for tag in sent_tags:
                    label_set.add(tag)
    return sorted(label_set)


#########################################
def _analyse_data(
    path: str,
    tokeniser: Tokeniser,
    log: Log,
) -> Tuple[Mapping[str, int], int]:
    '''
    Get the total number of rows and the maximum sentence length.

    :param path: The path to the raw dataset.
    :param tokeniser: The tokeniser to use.
    :param log: The log to pass progress information to.
    :return: A rows-len pair where rows is the total number of rows and len is
        the maximum sentence length.
    '''
    num_rows = {'train': 0, 'dev': 0, 'test': 0}
    max_len = 0
    splits_to_files = _get_files(path)
    i = 0
    num_files = sum(len(files) for (_, files) in splits_to_files.items())
    log.progress_start(0, num_files)
    for (split, files) in splits_to_files.items():
        for (fname, lang_name) in files:
            if split == 'train' and lang_name != 'en':
                continue
            for sent_words in _parse_conllu(fname)[0]:
                if not tokeniser.all_known_tokens(' '.join(sent_words)):
                    continue
                num_rows[split] += 1
                max_len = max(max_len, len(tokeniser.indexify_text(' '.join(sent_words))))

            i += 1
            log.progress_update(i)

    log.progress_end()
    return (num_rows, max_len)


#########################################
def _transfer_data(
    src_path: str,
    dst_path: str,
    spec: UDPOSDataSpec,
    num_rows: Mapping[str, int],
    log: Log,
) -> None:
    '''
    Transfer all the raw data into the dataset.

    :param src_path: The path to the raw dataset.
    :param dst_path: The path to store the processed dataset.
    :param spec: The data specification.
    :param num_rows: A mapping from data split to number of rows in the split.
    :param log: The log to pass progress information to.
    '''
    dset_train_orig_en = DatasetFile[UDPOSDataRow](
        os.path.join(dst_path, 'dataset_train-orig-en.hdf'), spec
    )
    dset_dev = DatasetFile[UDPOSDataRow](
        os.path.join(dst_path, 'dataset_dev.hdf'), spec
    )
    dset_test = DatasetFile[UDPOSDataRow](
        os.path.join(dst_path, 'dataset_test.hdf'), spec
    )

    with autocloser([dset_train_orig_en, dset_dev, dset_test]):
        log.log_message('> Creating train-orig-en dataset.')
        dset_train_orig_en.init(num_rows['train'])

        log.log_message('> Creating dev dataset.')
        dset_dev.init(num_rows['dev'])

        log.log_message('> Creating test dataset.')
        dset_test.init(num_rows['test'])

        dset_train_orig_en.load(as_readonly=False)
        dset_dev.load(as_readonly=False)
        dset_test.load(as_readonly=False)

        splits_to_files = _get_files(src_path)
        for (dset_split, dset_fname, dset_obj) in [
            ('train', 'train-orig-en', dset_train_orig_en),
            ('dev', 'dev', dset_dev),
            ('test', 'test', dset_test),
        ]:
            log.log_message('> Transferring data to {} dataset.'.format(dset_fname))
            row_num = 0
            i = 0
            log.progress_start(0, len(splits_to_files[dset_split]))
            for (fname, lang_name) in splits_to_files[dset_split]:
                if dset_split == 'train' and lang_name != 'en':
                    continue
                (sents_words, sents_tags) = _parse_conllu(fname)
                for (sent_words, sent_tags) in zip(sents_words, sents_tags):
                    if not spec.tokeniser.all_known_tokens(' '.join(sent_words)):
                        continue
                    row = UDPOSDataRow(sent_words, lang_name, sent_tags)
                    preprocessed = spec.preprocess(row)
                    dset_obj.set_row(row_num, preprocessed)
                    row_num += 1

                i += 1
                log.progress_update(i)

            log.progress_end()


#########################################
def _split_data(
    dst_path: str,
    spec: UDPOSDataSpec,
    train_fraction: float,
    val_fraction: float,
    rng: RandomNumberGenerator,
    log: Log,
) -> None:
    '''
    Take the monolitic dataset and split it into train-val-dev-test splits.

    :param dst_path: The path to store the processed dataset.
    :param spec: The data specification associated with the dataset.
    :param train_fraction: The fraction of the raw training set to use for the training set.
    :param val_fraction: The fraction of the raw training set to use for the validation set.
    :param rng: The random number generator to use.
    :param log: The log to receive progress information.
    '''
    dset_train_orig_en = DatasetFile[UDPOSDataRow](
        os.path.join(dst_path, 'dataset_train-orig-en.hdf'), spec
    )
    with autocloser([dset_train_orig_en]):
        dset_train_orig_en.init()
        dset_train_orig_en.load(as_readonly=True)

        def factory(
            name: str,
            size: int,
        ) -> Dataset[UDPOSDataRow]:
            sub_dataset = DatasetFile[UDPOSDataRow](
                os.path.join(dst_path, 'dataset_{}.hdf'.format(name)),
                spec,
            )
            sub_dataset.init(size)
            sub_dataset.load(as_readonly=False)
            return sub_dataset

        log.log_message('> Creating base split.')
        dset_train_orig_en.split(
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
    path: str,
    spec: UDPOSDataSpec,
    log: Log,
) -> None:
    '''
    Output a human readable version of the contents of the dataset.

    :param path: The path to the processed dataset folder.
    :param spec: The data specification associated with the dataset.
    :param log: The log to pass progress information to.
    '''
    dset_train_orig_en = DatasetFile[UDPOSDataRow](
        os.path.join(path, 'dataset_train-orig-en.hdf'), spec
    )
    dset_train = DatasetFile[UDPOSDataRow](
        os.path.join(path, 'dataset_train.hdf'), spec
    )
    dset_val = DatasetFile[UDPOSDataRow](
        os.path.join(path, 'dataset_val.hdf'), spec
    )
    dset_dev = DatasetFile[UDPOSDataRow](
        os.path.join(path, 'dataset_dev.hdf'), spec
    )
    dset_test = DatasetFile[UDPOSDataRow](
        os.path.join(path, 'dataset_test.hdf'), spec
    )

    with autocloser([
        dset_train_orig_en,
        dset_train,
        dset_val,
        dset_dev,
        dset_test,
    ]):
        dset_train_orig_en.init()
        dset_train.init()
        dset_val.init()
        dset_dev.init()
        dset_test.init()

        dset_train_orig_en.load(as_readonly=True)
        dset_train.load(as_readonly=True)
        dset_val.load(as_readonly=True)
        dset_dev.load(as_readonly=True)
        dset_test.load(as_readonly=True)

        for (dset_name, dset_obj) in [
            ('train-orig-en', dset_train_orig_en),
            ('train', dset_train),
            ('val', dset_val),
            ('dev', dset_dev),
            ('test', dset_test),
        ]:
            log.log_message('')
            log.log_message('Dataset: {}'.format(dset_name))

            if dset_name != 'train-orig-en':
                data_file = CsvFile(os.path.join(path, 'dataset_{}.csv'.format(dset_name)))
                data_file.init(
                    ['row', 'language', 'tokens', 'labels']
                )

            lang_freqs = {lang_name: 0 for lang_name in spec.lang_names}
            label_freqs = {label_name: 0 for label_name in spec.label_names}
            log.progress_start(0, dset_obj.size)
            for i in range(dset_obj.size):
                row = dset_obj.get_row(i)
                row_label_names = [
                    spec.label_names[label_index]
                    for label_index in row['labels'][
                        :row['num_tokens'] - spec.tokeniser.noncontent_total
                    ].tolist()
                ]
                row_label_freqs: Counter[str] = collections.Counter(row_label_names)
                row_lang_name = spec.lang_names[row['lang'].tolist()]
                if dset_name != 'train-orig-en':
                    data_file.append([
                        i,
                        row_lang_name,
                        ' '.join(
                            spec.tokeniser.textify_indexes(
                                row['tokens'][:row['num_tokens']][spec.tokeniser.content_slice]
                            )[0]
                        ),
                        ' '.join(row_label_names),
                    ])
                lang_freqs[row_lang_name] += 1
                for label in row_label_freqs:
                    label_freqs[label] += row_label_freqs[label]

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
def udpos_preprocess(
    src_path: str,
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
    Save the entire UDPOS dataset of files in preprocessed form.

    :param src_path: The path to the raw UDPOS dataset directory.
    :param dst_path: The path to the directory to contain the preprocessed dataset.
    :param tokeniser_name: The tokeniser to use.
    :param train_fraction: The fraction of the raw training set to use for the training set.
    :param val_fraction: The fraction of the raw training set to use for the validation set.
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
            log.log_message('Preprocessing UDPOS dataset.')
            log.log_message('')

            AboutFile(os.path.join(dst_path, 'about.csv')).init()
            VersionFile(os.path.join(dst_path, 'dataset_version.txt')).init(DATASET_VERSION)
            if seed is None:
                seed = RandomNumberGenerator.make_seed()

            log.log_message('Dataset version: {}'.format(DATASET_VERSION))
            log.log_message('Seed: {}'.format(seed))
            log.log_message('Source path: {}'.format(src_path))
            log.log_message('Destination path: {}'.format(dst_path))
            log.log_message('Tokeniser: {}'.format(tokeniser_name))
            log.log_message('Splits')
            log.log_message('> Train: {:.2%}'.format(train_fraction))
            log.log_message('> Val: {:.2%}'.format(val_fraction))
            log.log_message('Maximum number of tokens: {}'.format(max_num_tokens))
            log.log_message('')

            rng = RandomNumberGenerator(seed)

            log.log_message('Extracting label set.')
            label_names = _extract_label_set(src_path)

            log.log_message('Extracting language set.')
            lang_names = _extract_language_set(src_path)

            log.log_message('Creating data specification.')
            spec = UDPOSDataSpec(
                max_num_tokens, tokeniser_name, lang_names, label_names
            )

            with open(os.path.join(dst_path, 'lang_names.txt'), 'w', encoding='utf-8') as f_lang:
                print('\n'.join(spec.lang_names), file=f_lang)
            with open(os.path.join(dst_path, 'label_names.txt'), 'w', encoding='utf-8') as f_label:
                print('\n'.join(spec.label_names), file=f_label)
            with open(os.path.join(dst_path, 'data_spec.json'), 'w', encoding='utf-8') as f_spec:
                print(spec.to_json(), file=f_spec)

            log.log_message('Analysing data.')
            (num_rows, max_len) = _analyse_data(src_path, spec.tokeniser, log)
            log.log_message('> Longest sentence: {}'.format(max_len))
            log.log_message('')

            log.log_message('Transferring data.')
            _transfer_data(src_path, dst_path, spec, num_rows, log)
            log.log_message('')

            log.log_message('Splitting dataset.')
            _split_data(
                dst_path, spec,
                train_fraction, val_fraction,
                rng, log,
            )
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
