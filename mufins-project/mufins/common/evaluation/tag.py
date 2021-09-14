'''
Module for evaluating tagging tasks.
'''

import warnings
from typing import Sequence
import seqeval.metrics


#########################################
def eval_tags_f1(
    preds: Sequence[Sequence[str]],
    trues: Sequence[Sequence[str]],
    average: str,
) -> float:
    '''
    Use F1-score for evaluating a sequence of labelled sequences.

    :param preds: The predicted labels.
    :param trues: The true labels.
    :param average: Either micro or macro.
    :return: The F1-score as a fraction.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return seqeval.metrics.f1_score(
            preds, trues,
            average=average,
            zero_division=0,
        )
