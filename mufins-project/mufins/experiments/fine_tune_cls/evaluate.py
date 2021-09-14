'''
High-level evaluation for the model.
'''

import os
import json
from typing import Mapping, Tuple, Union, Callable, Sequence, Dict, Optional, List
import sklearn.manifold
import sklearn.metrics
import sklearn.cluster
import scipy.stats
import numpy as np
from mufins.common.random.random_number_generator import RandomNumberGenerator
from mufins.common.dataset.dataset import Dataset
from mufins.common.log.log import Log
from mufins.common.file.csv_file import CsvFile
from mufins.common.evaluation.cls import eval_cls_f1, eval_cls_acc
from mufins.common.chart.legend import get_legend
from mufins.common.chart.scatter import get_scatter
from mufins.dataprocs.xnli.data_row import XNLIDataRow
from mufins.dataprocs.wikipedia.data_row import WikipediaDataRow
from mufins.dataprocs.xnli.data_spec import XNLIDataSpec
from mufins.dataprocs.wikipedia.data_spec import WikipediaDataSpec


#########################################
LabelEncoderType = Callable[
    [
        Dataset[XNLIDataRow], # dset_label
        int, # batch_size
        Optional[Log], # log
        Optional[str], # label
        Optional[RandomNumberGenerator], # sampler
        Optional[int], # sample_size_per_unit
    ],
    Tuple[
        Sequence[Sequence[float]],
        Sequence[str],
        Sequence[str],
    ]
]

LangEncoderType = Callable[
    [
        Dataset[WikipediaDataRow], # dset_lang
        int, # batch_size
        Optional[Log], # log
        Optional[str], # label
        Optional[RandomNumberGenerator], # sampler
        Optional[int], # sample_size_per_unit
    ],
    Tuple[
        Sequence[Sequence[float]],
        Sequence[str],
    ]
]

LabelPredictorType = Callable[
    [
        Dataset[XNLIDataRow], # dset_label
        int, # batch_size
        Optional[Log], # log
        Optional[str], # label
    ],
    Tuple[
        Sequence[Sequence[float]],
        Sequence[str],
        Sequence[str],
    ]
]

LangPredictorType = Callable[
    [
        Union[Dataset[WikipediaDataRow], Dataset[XNLIDataRow]], # dset
        int, # batch_size
        Optional[Log], # log
        Optional[str], # label
    ],
    Tuple[
        Sequence[Sequence[float]],
        Sequence[str],
        Sequence[str],
    ]
]


#########################################
def prepare_encoding_plot(
    path: str,
    label_spec: XNLIDataSpec,
    lang_spec: WikipediaDataSpec,
) -> None:
    '''
    Prepare the directories and legends for the encoding plot.

    :param path: The path in which to create the directories.
    :param label_spec: The label data specification.
    :param lang_spec: The language data specification.
    '''
    os.makedirs(os.path.join(path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(path, 'lang'), exist_ok=True)

    with get_legend(lang_spec.lang_names) as fig:
        fig.savefig(
            os.path.join(path, 'legend_lang.pdf')
        )
    with get_legend(label_spec.label_names) as fig:
        fig.savefig(
            os.path.join(path, 'legend_label.pdf')
        )


#########################################
def encoding_plot_label(
    path: str,
    label_spec: XNLIDataSpec,
    lang_spec: WikipediaDataSpec,
    dset_label: Dataset[XNLIDataRow],
    label_encoder: LabelEncoderType,
    batch_size: int,
    log: Log,
    fname_prefix: str = '',
) -> Tuple[float, float]:
    '''
    Save encoding plots for label data.

    :param path: Path used in `prepare_encoding_plot`.
    :param label_spec: The label data specification.
    :param lang_spec: The language data specification.
    :param dset_label: The label data set to use.
    :param label_encoder: A function that encodes label texts into vectors.
    :param batch_size: The maximum number of data items to process at once.
    :param log: The log.
    :param fname_prefix: A prefix to put in front of the created files' file names.
    :return: The pair of v-measures of the cluster-label and cluster-language agreement.
    '''
    rng = RandomNumberGenerator(0)

    (encodings, lang_trues, label_trues) = label_encoder(
        dset_label, batch_size, log, 'Encoding labels', rng.get_child(), 50
    )
    plot_csv = CsvFile(
        os.path.join(path, 'label', '{}plot_fullenc.csv'.format(fname_prefix))
    )
    plot_csv.init(['encoding', 'lang', 'label'], clear=True)
    for (enc, lang, label) in zip(encodings, lang_trues, label_trues):
        plot_csv.append([json.dumps(enc), lang, label])

    log.progress_start(0, 10, 'Measuring cluster-label agreement')
    lang_vmeasures: List[float] = []
    label_vmeasures: List[float] = []
    for i in range(1, 10+1):
        clusters = sklearn.cluster.KMeans(
            n_clusters=min(len(lang_spec.lang_names), len(encodings)), random_state=i,
            n_init=1, max_iter=100, init='random', algorithm='full',
        ).fit_predict(encodings)
        lang_vmeasures.append(sklearn.metrics.v_measure_score(lang_trues, clusters))

        clusters = sklearn.cluster.KMeans(
            n_clusters=min(len(label_spec.label_names), len(encodings)), random_state=i,
            n_init=1, max_iter=100, init='random', algorithm='full',
        ).fit_predict(encodings)
        label_vmeasures.append(sklearn.metrics.v_measure_score(label_trues, clusters))

        log.progress_update(i)
    log.progress_end()

    log.progress_start(0, 1, 'Plotting labels')
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0, init='pca')
    encodings = tsne.fit_transform(encodings).tolist()
    log.progress_update(1)
    log.progress_end()

    with get_scatter(
        label_spec.label_names,
        [x for [x, y] in encodings],
        [y for [x, y] in encodings],
        label_trues,
    ) as fig:
        fig.savefig(
            os.path.join(path, 'label', '{}label_plot.pdf'.format(fname_prefix))
        )
    with get_scatter(
        label_spec.label_names,
        [x for (lang, [x, y]) in zip(lang_trues, encodings) if lang != 'en'],
        [y for (lang, [x, y]) in zip(lang_trues, encodings) if lang != 'en'],
        [label for (lang, label) in zip(lang_trues, label_trues) if lang != 'en'],
    ) as fig:
        fig.savefig(
            os.path.join(path, 'label', '{}label_plot_no-en.pdf'.format(fname_prefix))
        )

    with get_scatter(
        lang_spec.lang_names,
        [x for [x, y] in encodings],
        [y for [x, y] in encodings],
        lang_trues,
    ) as fig:
        fig.savefig(
            os.path.join(path, 'label', '{}lang_plot.pdf'.format(fname_prefix))
        )

    plot_csv = CsvFile(
        os.path.join(path, 'label', '{}plot.csv'.format(fname_prefix))
    )
    plot_csv.init(['x', 'y', 'lang', 'label'], clear=True)
    for ([x, y], lang, label) in zip(encodings, lang_trues, label_trues):
        plot_csv.append([x, y, lang, label])

    return (np.mean(label_vmeasures).tolist(), np.mean(lang_vmeasures).tolist())


#########################################
def encoding_plot_lang(
    path: str,
    lang_spec: WikipediaDataSpec,
    dset_lang: Dataset[WikipediaDataRow],
    lang_encoder: LangEncoderType,
    batch_size: int,
    log: Log,
    fname_prefix: str = '',
) -> float:
    '''
    Save encoding plots for language data.

    :param path: Path used in `prepare_encoding_plot`.
    :param lang_spec: The language data specification.
    :param dset_lang: The language data set to use.
    :param lang_encoder: A function that encodes language texts into vectors.
    :param batch_size: The maximum number of data items to process at once.
    :param log: The log.
    :param fname_prefix: A prefix to put in front of the created files' file names.
    :return: The v-measures of the cluster-language agreement.
    '''
    rng = RandomNumberGenerator(0)

    (encodings, lang_trues) = lang_encoder(
        dset_lang, batch_size, log, 'Encoding langs.', rng.get_child(), 100
    )
    plot_csv = CsvFile(
        os.path.join(path, 'lang', '{}plot_fullenc.csv'.format(fname_prefix))
    )
    plot_csv.init(['encoding', 'lang'], clear=True)
    for (enc, lang) in zip(encodings, lang_trues):
        plot_csv.append([json.dumps(enc), lang])

    log.progress_start(0, 10, 'Measuring cluster-lang agreement')
    lang_vmeasures: List[float] = []
    for i in range(1, 10+1):
        clusters = sklearn.cluster.KMeans(
            n_clusters=min(len(lang_spec.lang_names), len(encodings)), random_state=i,
            n_init=1, max_iter=100, init='random', algorithm='full',
        ).fit_predict(encodings)
        lang_vmeasures.append(sklearn.metrics.v_measure_score(lang_trues, clusters))

        log.progress_update(i)
    log.progress_end()

    log.progress_start(0, 1, 'Plotting langs.')
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0, init='pca')
    encodings = tsne.fit_transform(encodings).tolist()
    log.progress_update(1)
    log.progress_end()

    with get_scatter(
        lang_spec.lang_names,
        [x for [x, y] in encodings],
        [y for [x, y] in encodings],
        lang_trues,
    ) as fig:
        fig.savefig(
            os.path.join(path, 'lang', '{}plot.pdf'.format(fname_prefix))
        )
    plot_csv = CsvFile(
        os.path.join(path, 'lang', '{}plot.csv'.format(fname_prefix))
    )
    plot_csv.init(['x', 'y', 'lang'], clear=True)
    for ([x, y], lang) in zip(encodings, lang_trues):
        plot_csv.append([x, y, lang])

    return np.mean(lang_vmeasures).tolist()


#########################################
def eval_label(
    label_spec: XNLIDataSpec,
    lang_spec: WikipediaDataSpec,
    dset_label: Dataset[XNLIDataRow],
    label_predictor: LabelPredictorType,
    lang_predictor: LangPredictorType,
    hyperparameter_search_mode: bool,
    batch_size: int,
    log: Log,
    path: Optional[str] = None,
    fname_prefix: str = '',
) -> Tuple[
    float,
    float,
    Mapping[str, Optional[float]],
    float,
]:
    '''
    Return label-based evaluation information.

    :param label_spec: The label data specification.
    :param lang_spec: The language data specification.
    :param dset_label: The label data set to use.
    :param label_predictor: A function that predicts labels.
    :param lang_predictor: A function that predicts languages.
    :param hyperparameter_search_mode: Whether to enter into hyperparameter search mode where
        minimal output and evaluation is produced.
    :param batch_size: The maximum number of data items to process at once.
    :param log: The log.
    :param path: The path to the folder in which to save the outputs file.
        If None, then no outputs file is saved.
    :param fname_prefix: A prefix to put in front of the created file's file name.
    :return: A 4-tuple consisting of the following:

        - The label macro F1 score.
        - The label accuracy score.
        - A dictionary mapping each language to its individual macro F1 score.
        - The language macro F1 score.
    '''
    label_score_f1_macro = 0.0
    label_score_acc = 0.0
    label_score_f1_macro_by_lang: Dict[str, Optional[float]] = dict()
    lang_score_f1_macro = 0.0

    (
        label_logprobs, label_preds, label_trues
    ) = label_predictor(
        dset_label, batch_size, log, 'Eval. label label'
    )

    if not hyperparameter_search_mode:
        (
            lang_logprobs, lang_preds, lang_trues
        ) = lang_predictor(
            dset_label, batch_size, log, 'Eval. label lang.'
        )

        if path is not None:
            outputs_file = CsvFile(
                os.path.join(path, '{}label_outputs.csv'.format(fname_prefix)),
            )
            outputs_file.init(
                [
                    'sent_index',
                    'text',
                    'true_lang',
                    'pred_lang',
                    'true_label',
                    'pred_label',
                ] + [
                    'logprob_lang_{}'.format(lang) for lang in lang_spec.lang_names
                ] + [
                    'logprob_label_{}'.format(label) for label in label_spec.label_names
                ],
                clear=True,
            )
            for sent_index in range(dset_label.size):
                row = dset_label.get_row(sent_index)

                tokens = label_spec.tokeniser.textify_indexes(
                    row['tokens'][:row['num_tokens']][label_spec.tokeniser.content_slice].tolist()
                )[0]

                outputs_file.append(
                    [
                        sent_index,
                        ' '.join(tokens),
                        lang_trues[sent_index],
                        lang_preds[sent_index],
                        label_trues[sent_index],
                        label_preds[sent_index],
                    ] + [
                        '{:.10f}'.format(p) for p in lang_logprobs[sent_index]
                    ] + [
                        '{:.10f}'.format(p) for p in label_logprobs[sent_index]
                    ]
                )

    label_score_f1_macro = eval_cls_f1(
        label_preds, label_trues,
        average='macro',
    )

    if not hyperparameter_search_mode:
        label_score_acc = eval_cls_acc(
            label_preds, label_trues,
        )

        label_score_f1_macro_by_lang = dict()
        for lang in label_spec.lang_names:
            filtered_label_preds = [
                sent_label
                for (sent_label, sent_lang) in zip(label_preds, lang_trues)
                if sent_lang == lang
            ]
            if len(filtered_label_preds) == 0:
                label_score_f1_macro_by_lang[lang] = None
                continue
            filtered_label_trues = [
                sent_label
                for (sent_label, sent_lang) in zip(label_trues, lang_trues)
                if sent_lang == lang
            ]
            label_score_f1_macro_by_lang[lang] = eval_cls_f1(
                filtered_label_preds, filtered_label_trues,
                average='macro',
            )

        lang_score_f1_macro = eval_cls_f1(
            lang_preds, lang_trues,
            average='macro',
        )

    return (
        label_score_f1_macro,
        label_score_acc,
        label_score_f1_macro_by_lang,
        lang_score_f1_macro,
    )


#########################################
def eval_lang(
    lang_spec: WikipediaDataSpec,
    dset_lang: Dataset[WikipediaDataRow],
    lang_predictor: LangPredictorType,
    batch_size: int,
    log: Log,
    path: Optional[str] = None,
    fname_prefix: str = '',
) -> Tuple[
    float, float, float, float
]:
    '''
    Return language-based evaluation information.

    :param lang_spec: The language data specification.
    :param dset_lang: The language data set to use.
    :param lang_predictor: A function that predicts languages.
    :param batch_size: The maximum number of data items to process at once.
    :param log: The log.
    :param path: The path to the folder in which to save the outputs file.
        If None, then no outputs file is saved.
    :param fname_prefix: A prefix to put in front of the created file's file name.

    :return: A 4-tuple consisting of the following:

        - The macro F1 score.
        - The global minimum entropy of all the texts' language probabilities.
        - The global mean entropy of all the texts' language probabilities.
        - The global maximum entropy of all the texts' language probabilities.
    '''
    (
        logprobs, preds, trues
    ) = lang_predictor(
        dset_lang, batch_size, log, 'Eval lang.'
    )

    if path is not None:
        outputs_file = CsvFile(
            os.path.join(path, '{}lang_outputs.csv'.format(fname_prefix)),
        )
        outputs_file.init(
            [
                'sent_index',
                'text',
                'true_lang',
                'pred_lang',
            ] + [
                'logprob_lang_{}'.format(lang) for lang in lang_spec.lang_names
            ],
            clear=True,
        )
        for sent_index in range(dset_lang.size):
            row = dset_lang.get_row(sent_index)

            tokens = lang_spec.tokeniser.textify_indexes(
                row['tokens'][:row['num_tokens']][lang_spec.tokeniser.content_slice].tolist()
            )[0]

            outputs_file.append(
                [
                    sent_index,
                    ' '.join(tokens),
                    trues[sent_index],
                    preds[sent_index],
                ] + [
                    '{:.10f}'.format(p) for p in logprobs[sent_index]
                ]
            )

    score_f1_macro = eval_cls_f1(
        preds, trues,
        average='macro',
    )

    prob_entropies = scipy.stats.entropy(
        np.exp(logprobs), axis=1,
    )

    return (
        score_f1_macro,
        np.min(prob_entropies),
        np.mean(prob_entropies),
        np.max(prob_entropies),
    )
