'''
Mock training process class.
'''

from typing import Iterator, Optional, Sequence, Mapping
import numpy as np
from mufins.common.model.training_process_standard import TrainingProcessStandard
from mufins.tests.common.model.model.mock_model import MockModel


#########################################
class Interruption(Exception):
    '''
    Interruption simulation exception.
    '''


#########################################
class MockTrainingProcess(TrainingProcessStandard):
    '''
    A mock implementation of the training process class which simulates an interruption on a
    given epoch number.
    '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=no-self-use
    # pylint: disable=unused-argument
    # pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        expected_first_epoch: Optional[int],
        interrupt_on: Optional[int],
        model: MockModel,
    ) -> None:
        '''
        Constructor.

        :param expected_first_epoch: The epoch number expected to start from.
            If None then do not check anything.
        :param interrupt_on: The epoch number on which to interrupt.
            If None then do not interrupt.
        :param model: The model being trained.
        '''
        super().__init__()
        self.interrupt_on: Optional[int] = interrupt_on
        self.expected_first_epoch: Optional[int] = expected_first_epoch
        self.model: MockModel = model

    #########################################
    def _get_minibatches(
        self,
        epoch_num: int,
        minibatch_size: int,
    ) -> Iterator[Sequence[Mapping[str, np.ndarray]]]:
        '''
        Get an iterator over all the batches in the training set.

        These batches will be passed to `batch_fit` in the backend model.

        :param epoch_num: The epoch number about to start.
        :param minibatch_size: The number of items in a minibatch.
        :return: An iterator of minibatchbatches.
        '''
        data = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
        ], np.float32)
        for i in range(data.shape[0]//minibatch_size):
            yield ({'x': data[i*minibatch_size:(i+1)*minibatch_size]},)

    #########################################
    def _on_epoch_start(
        self,
        epoch_num: int,
    ) -> None:
        '''
        Listener for when an epoch starts.

        :param epoch_num: The epoch number about to start.
        '''
        if self.expected_first_epoch is not None:
            assert epoch_num == self.expected_first_epoch, (
                'First epoch was {} instead of {}.'.format(
                    epoch_num, self.expected_first_epoch
                )
            )
            self.expected_first_epoch = None

        if self.interrupt_on is not None and epoch_num == self.interrupt_on:
            raise Interruption()

    #########################################
    def _get_val_score(
        self,
        epoch_num: int,
    ) -> float:
        '''
        Get the validation score using the current model (score is to be maximised).

        :param epoch_num: The current epoch number.
        :return: The score.
        '''
        return float(epoch_num)
