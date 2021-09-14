'''
Main data preprocessor for the Wikipedia dataset.
'''

import os
from typing import List, Sequence, Tuple, cast, Optional
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
from mufins.dataprocs.wikipedia.data_spec import WikipediaDataSpec
from mufins.dataprocs.wikipedia.data_row import WikipediaDataRow


#########################################
DATASET_VERSION = 2


#########################################
def _extract_language_set(
    src_path: str,
) -> Sequence[str]:
    '''
    Get all the different languages used in the dataset.

    :param src_path: The path to the raw Wikipedia dataset.
    :return: A list of unique language names.
    '''
    return sorted({
        fname for fname in os.listdir(src_path)
        if os.path.isdir(os.path.join(src_path, fname))
    } & STD_LANGS)


#########################################
def _analyse_data(
    src_path: str,
    dst_path: str,
    tokeniser: Tokeniser,
    max_num_texts_per_lang: int,
    min_num_chars: int,
    rng: RandomNumberGenerator,
    log: Log,
) -> Tuple[int, int]:
    '''
    Save a text file with a random subset of valid texts and return the total number of rows and
        the maximum text length.

    :param src_path: The path to the raw Wikipedia dataset.
    :param dst_path: The path to store the processed dataset.
    :param tokeniser: The tokeniser to use.
    :param max_num_texts_per_lang: The maximum number of texts per language to include in the
        data set.
    :param min_num_chars: The minimum number of characters in a text to be allowed in.
    :param rng: The random number generator to use.
    :param log: The log to pass progress information to.
    :return: A pair consisting of the number of texts in the data and the maximum number of tokens
        in a text.
    '''
    langs = _extract_language_set(src_path)
    num_rows = 0
    max_len = 0
    with open(os.path.join(dst_path, 'subset.txt'), 'w', encoding='utf-8') as f_out:
        for lang in langs:
            log.log_message('> {}'.format(lang))
            log.log_message('>> Filtering')
            valid_lines = []
            total_num_files = sum(
                len(os.listdir(os.path.join(src_path, lang, folder)))
                for folder in os.listdir(os.path.join(src_path, lang))
            )
            i = 0
            log.progress_start(0, total_num_files)
            for (folder_idx, folder) in enumerate(
                sorted(os.listdir(os.path.join(src_path, lang)))
            ):
                for (fname_idx, fname) in enumerate(
                    sorted(os.listdir(os.path.join(src_path, lang, folder)))
                ):
                    with open(
                        os.path.join(src_path, lang, folder, fname), 'r', encoding='utf-8'
                    ) as f_in:
                        for (line_idx, line) in enumerate(f_in):
                            line = line.strip()
                            if line == '':
                                continue
                            if line.startswith('<doc '):
                                continue
                            if line == '</doc>':
                                continue
                            if len(line) < min_num_chars:
                                continue
                            if not tokeniser.all_known_tokens(line):
                                continue
                            valid_lines.append((folder_idx, fname_idx, line_idx))

                    i += 1
                    log.progress_update(i)
            log.progress_end()

            if len(valid_lines) == 0:
                raise ValueError('All texts have been filtered out. Relax the criteria.')

            log.log_message('>> Choosing subset')
            rng.shuffle(valid_lines)
            valid_lines = valid_lines[:max_num_texts_per_lang]
            valid_lines.sort(reverse=True)
            log.log_message('>>> {}'.format(len(valid_lines)))

            log.log_message('>> Copying to subset.txt')
            for (folder_idx, folder) in enumerate(
                sorted(os.listdir(os.path.join(src_path, lang)))
            ):
                if folder_idx != valid_lines[-1][0]:
                    continue
                for (fname_idx, fname) in enumerate(
                    sorted(os.listdir(os.path.join(src_path, lang, folder)))
                ):
                    if fname_idx != valid_lines[-1][1]:
                        continue
                    with open(
                        os.path.join(src_path, lang, folder, fname), 'r', encoding='utf-8'
                    ) as f_in:
                        for (line_idx, line) in enumerate(f_in):
                            if line_idx != valid_lines[-1][2]:
                                continue
                            assert (folder_idx, fname_idx, line_idx) == valid_lines[-1]

                            line = line.strip().replace('\t', ' ')
                            print(line, lang, sep='\t', file=f_out)

                            text_len = len(tokeniser.indexify_text(line))

                            max_len = max(max_len, text_len)
                            num_rows += 1
                            valid_lines.pop()
                            if len(valid_lines) == 0:
                                break
                            if (folder_idx, fname_idx) != valid_lines[-1][:2]:
                                break
                    if len(valid_lines) == 0:
                        break
                    if folder_idx != valid_lines[-1][0]:
                        break
                if len(valid_lines) == 0:
                    break

    return (num_rows, max_len)


#########################################
def _transfer_data(
    dst_path: str,
    spec: WikipediaDataSpec,
    num_rows: int,
    log: Log,
) -> None:
    '''
    Transfer all the selected raw data into the dataset.

    :param dst_path: The path to store the processed dataset.
    :param spec: The data specification.
    :param num_rows: The number of rows in the data.
    :param log: The log to pass progress information to.
    '''
    dset = DatasetFile[WikipediaDataRow](
        os.path.join(dst_path, 'dataset.hdf'), spec
    )

    with autocloser([dset]):
        log.log_message('> Creating dataset.')
        dset.init(num_rows)

        dset.load(as_readonly=False)

        log.log_message('> Transferring data to dataset.')
        row_num = 0
        log.progress_start(0, num_rows)
        with open(os.path.join(dst_path, 'subset.txt'), 'r', encoding='utf-8') as f:
            for (row_num, line) in enumerate(f):
                line = line.strip()
                if line == '':
                    continue

                (text, lang_name) = line.split('\t')

                row = WikipediaDataRow(
                    text,
                    lang_name,
                )
                preprocessed = dset.spec.preprocess(row)
                dset.set_row(row_num, preprocessed)

                log.progress_update(row_num + 1)

        log.progress_end()


#########################################
def _split_data(
    dst_path: str,
    spec: WikipediaDataSpec,
    train_fraction: float,
    val_fraction: float,
    dev_fraction: float,
    test_fraction: float,
    rng: RandomNumberGenerator,
    log: Log,
) -> None:
    '''
    Take the data set and split it into train-val-dev-test splits.

    :param dst_path: The path to store the processed dataset.
    :param spec: The data specification associated with the dataset.
    :param train_fraction: The fraction of the data set to use for the train set.
    :param val_fraction: The fraction of the data set to use for the validation set.
    :param dev_fraction: The fraction of the data set to use for the development set.
    :param test_fraction: The fraction of the data set to use for the development set.
    :param rng: The random number generator to use.
    :param log: The log to receive progress information.
    '''
    dset = DatasetFile[WikipediaDataRow](
        os.path.join(dst_path, 'dataset.hdf'), spec
    )
    with autocloser([dset]):
        dset.init()
        dset.load(as_readonly=True)

        def factory(
            name: str,
            size: int,
        ) -> Dataset[WikipediaDataRow]:
            sub_dataset = DatasetFile[WikipediaDataRow](
                os.path.join(dst_path, 'dataset_{}.hdf'.format(name)),
                spec,
            )
            sub_dataset.init(size)
            sub_dataset.load(as_readonly=False)
            return sub_dataset

        log.log_message('> Creating base split.')
        dset.split(
            partition_fractions=[
                ('train', train_fraction),
                ('val', val_fraction),
                ('dev', dev_fraction),
                ('test', test_fraction),
            ],
            dataset_factory=factory,
            stratification_key=lambda row:row['lang'].tolist(),
            rng=rng,
            log=log,
        )


#########################################
def _output_readable_dataset(
    dst_path: str,
    spec: WikipediaDataSpec,
    log: Log,
) -> None:
    '''
    Output a human readable version of the contents of the dataset.

    :param dst_path: The path to the processed dataset folder.
    :param spec: The data specification associated with the dataset.
    :param log: The log to pass progress information to.
    '''
    dset = DatasetFile[WikipediaDataRow](
        os.path.join(dst_path, 'dataset.hdf'), spec
    )
    dset_train = DatasetFile[WikipediaDataRow](
        os.path.join(dst_path, 'dataset_train.hdf'), spec
    )
    dset_val = DatasetFile[WikipediaDataRow](
        os.path.join(dst_path, 'dataset_val.hdf'), spec
    )
    dset_dev = DatasetFile[WikipediaDataRow](
        os.path.join(dst_path, 'dataset_dev.hdf'), spec
    )
    dset_test = DatasetFile[WikipediaDataRow](
        os.path.join(dst_path, 'dataset_test.hdf'), spec
    )

    with autocloser([
        dset,
        dset_train,
        dset_val,
        dset_dev,
        dset_test,
    ]):
        dset.init()
        dset_train.init()
        dset_val.init()
        dset_dev.init()
        dset_test.init()

        dset.load(as_readonly=True)
        dset_train.load(as_readonly=True)
        dset_val.load(as_readonly=True)
        dset_dev.load(as_readonly=True)
        dset_test.load(as_readonly=True)

        for (dset_name, dset_obj) in [
            ('full', dset),
            ('train', dset_train),
            ('val', dset_val),
            ('dev', dset_dev),
            ('test', dset_test),
        ]:
            log.log_message('')
            log.log_message('Dataset: {}'.format(dset_name))

            if dset_name != 'full':
                data_file = CsvFile(os.path.join(dst_path, 'dataset_{}.csv'.format(dset_name)))
                data_file.init(
                    ['row', 'tokens', 'language']
                )

            lang_freqs = {lang_name: 0 for lang_name in spec.lang_names}
            log.progress_start(0, dset_obj.size)
            for i in range(dset_obj.size):
                row = dset_obj.get_row(i)
                row_lang_name = spec.lang_names[row['lang'].tolist()]
                if dset_name != 'full':
                    data_file.append([
                        i,
                        ' '.join(
                            spec.tokeniser.textify_indexes(
                                row['tokens'][:row['num_tokens']][spec.tokeniser.content_slice]
                            )[0]
                        ),
                        row_lang_name,
                    ])
                lang_freqs[row_lang_name] += 1

                log.progress_update(i)

            log.progress_end()

            log.log_message('Number of rows: {}'.format(dset_obj.size))
            log.log_message('Language frequencies:')
            for lang_name in spec.lang_names:
                log.log_message('> {}: {}'.format(
                    lang_name, lang_freqs[lang_name]
                ))


#########################################
def wikipedia_preprocess(
    src_path: str,
    dst_path: str,
    tokeniser_name: str,
    train_fraction: float,
    val_fraction: float,
    dev_fraction: float,
    test_fraction: float,
    max_num_texts_per_lang: int,
    min_num_chars: int,
    max_num_tokens: int,
    seed: Optional[int],
    verbose: bool = True,
    debug_mode: bool = True,
) -> None:
    '''
    Save the entire Wikipedia dataset of files in preprocessed form.

    :param src_path: The path to the raw Wikipedia dataset.
    :param dst_path: The path to the directory to contain the preprocessed dataset.
    :param tokeniser_name: The tokeniser to use.
    :param train_fraction: The fraction of the data to use for the training set.
    :param val_fraction: The fraction of the data to use for the validation set.
    :param dev_fraction: The fraction of the data to use for the development set.
    :param test_fraction: The fraction of the data to use for the test set.
    :param max_num_texts_per_lang: The maximum number of texts to keep from each language, which
        will then be split into train, val, dev, and test.
    :param min_num_chars: The minimum number of characters in a text to be allowed in.
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
            log.log_message('Preprocessing Wikipedia dataset.')
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
            log.log_message('> Dev: {:.2%}'.format(dev_fraction))
            log.log_message('> Test: {:.2%}'.format(test_fraction))
            log.log_message('Maximimum number of texts per languages: {}'.format(
                max_num_texts_per_lang
            ))
            log.log_message('Minimum number of characters: {}'.format(min_num_chars))
            log.log_message('Maximum number of tokens: {}'.format(max_num_tokens))
            log.log_message('')

            rng = RandomNumberGenerator(seed)

            log.log_message('Extracting language set.')
            lang_names = _extract_language_set(src_path)

            log.log_message('Creating data specification.')
            spec = WikipediaDataSpec(
                max_num_tokens, tokeniser_name, lang_names
            )

            with open(os.path.join(dst_path, 'lang_names.txt'), 'w', encoding='utf-8') as f_lang:
                print('\n'.join(spec.lang_names), file=f_lang)
            with open(os.path.join(dst_path, 'data_spec.json'), 'w', encoding='utf-8') as f_spec:
                print(spec.to_json(), file=f_spec)

            log.log_message('Analysing data.')
            (num_rows, max_len) = _analyse_data(
                src_path, dst_path, spec.tokeniser, max_num_texts_per_lang, min_num_chars,
                rng.get_child(), log
            )
            log.log_message('> Longest text: {}'.format(max_len))
            log.log_message('')

            log.log_message('Transferring data.')
            _transfer_data(dst_path, spec, num_rows, log)
            log.log_message('')

            log.log_message('Splitting dataset.')
            _split_data(
                dst_path, spec, train_fraction, val_fraction, dev_fraction, test_fraction,
                rng.get_child(), log
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
