'''
Standard abstract training process class.
'''

from typing import Optional, Sequence
from mufins.common.log.log import Log
from mufins.common.model.model_standard import ModelStandard
from mufins.common.model.training_process import TrainingProcess


#########################################

class TrainingProcessStandard(TrainingProcess[ModelStandard]):
    '''
    Provide a method for training a backend model using early stopping.
    '''
    #pylint: disable=too-few-public-methods

    #########################################
    def _train_epoch(
        self,
        epoch_num: int,
        minibatch_size: int,
        num_minibatches: Sequence[int],
        batch_size: int,
        model: ModelStandard,
        log: Optional[Log],
    ) -> float:
        '''
        Train the model for a single epoch and return the validation score.

        :param epoch_num: The epoch number about to start.
        :param minibatch_size: The number of items in a minibatch.
        :param num_minibatches: The number of minibatches in each training set.
        :param batch_size: The maximum amount of training items to process in one go.
        :param model: The backend model being trained.
        :param log: The log to monitor training progress.
        :return: The validation score.
        '''
        if log is not None:
            log.progress_start(
                0, max(num_minibatches), 'Epoch {}'.format(epoch_num)
            )

        for (minibatch_num, minibatch) in enumerate(
            self._get_minibatches(epoch_num, minibatch_size),
            1,
        ):
            model.batch_fit(minibatch, batch_size)

            if log is not None:
                log.progress_update(minibatch_num)
            self._on_minibatch_end(epoch_num, minibatch_num)

        if log is not None:
            log.progress_end()

        val_score = self._get_val_score(epoch_num)
        return val_score
