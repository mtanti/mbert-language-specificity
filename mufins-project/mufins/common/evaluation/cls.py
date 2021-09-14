'''
Module for evaluating classification tasks.
'''

from typing import Sequence
import sklearn.metrics


#########################################
def eval_cls_acc(
    preds: Sequence[str],
    trues: Sequence[str],
) -> float:
    '''
    Use accuracy for evaluating a sequence of classifications.

    :param preds: The predicted labels.
    :param trues: The true labels.
    :return: The accuracy as a fraction.
    '''
    return sklearn.metrics.accuracy_score(
        preds, trues,
    )

#########################################
def eval_cls_f1(
    preds: Sequence[str],
    trues: Sequence[str],
    average: str,
) -> float:
    '''
    Use F1-score for evaluating a sequence of classifications.

    :param preds: The predicted labels.
    :param trues: The true labels.
    :param average: Either micro or macro.
    :return: The F1-score as a fraction.
    '''
    return sklearn.metrics.f1_score(
        preds, trues,
        average=average,
        zero_division=0,
    )
