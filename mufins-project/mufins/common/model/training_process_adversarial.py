'''
Adversarial abstract training process class.

Like normal training process, but each epoch consists of training the model once on data set
and once on another data set, with the aim being to create adversarial learning.
This means that the discriminator part of the model is trained on one data set whilst the rest of
the model is trained on the other data set.

See documentation for `training_process`.
'''

from abc import abstractmethod
from typing import Iterator, Optional, Sequence, Mapping
import numpy as np
from mufins.common.log.log import Log
from mufins.common.model.model_adversarial import ModelAdversarial
from mufins.common.model.training_process import TrainingProcess


#########################################
class TrainingProcessAdversarial(TrainingProcess[ModelAdversarial]):
    '''
    Provide a method for training an adversarial backend model using early stopping.
    '''
    #pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        use_discriminator: bool = True
    ) -> None:
        '''
        Constructor.

        :param use_discriminator: Whether to use the discriminator or to treat training like a
            standard training process.
        '''
        super().__init__()
        self.use_discriminator: bool = use_discriminator

    #########################################
    def _train_epoch(
        self,
        epoch_num: int,
        minibatch_size: int,
        num_minibatches: Sequence[int],
        batch_size: int,
        model: ModelAdversarial,
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
        (num_minibatches_disc, num_minibatches_main) = num_minibatches

        if self.use_discriminator:
            if log is not None:
                log.progress_start(
                    0, num_minibatches_disc, 'Epoch {} (disc.)'.format(epoch_num)
                )

            model.set_curr_opt_index(0) # Discriminator

            for (minibatch_num, minibatch) in enumerate(
                self._get_minibatches_disc(epoch_num, minibatch_size),
                1,
            ):
                model.batch_fit(minibatch, batch_size)

                if log is not None:
                    log.progress_update(minibatch_num)
                self._on_minibatch_end_disc(epoch_num, minibatch_num)

            if log is not None:
                log.progress_end()

            self._on_discriminator_trained(epoch_num)

            model.set_curr_opt_index(1) # Main
            # Current optimiser is not set when training in standard mode, just like in standard
            # mode.

        if log is not None:
            log.progress_start(
                0, num_minibatches_main, 'Epoch {}{}'.format(
                    epoch_num,
                    ' (main)' if self.use_discriminator else ''
                )
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

    #########################################
    @abstractmethod
    def _get_minibatches_disc(
        self,
        epoch_num: int,
        minibatch_size: int,
    ) -> Iterator[Sequence[Mapping[str, np.ndarray]]]:
        '''
        Get an iterator over all the batches in the discriminator's training set.

        These batches will be passed to `batch_fit` in the backend model.

        :param epoch_num: The epoch number about to start.
        :param minibatch_size: The number of items in a minibatch.
        :return: An iterator of minibatchbatches.
        '''

    #########################################
    def _on_minibatch_end_disc(
        self,
        epoch_num: int,
        minibatch_num: int,
    ) -> None:
        '''
        Listener for when a discriminator's minibatch ends.

        :param epoch_num: The current epoch number.
        :param minibatch_num: The minibatch number just finished.
        '''

    #########################################
    def _on_discriminator_trained(
        self,
        epoch_num: int,
    ) -> None:
        '''
        Listener for when the disciminator has been trained, before training the rest of the model.

        :param epoch_num: The epoch number about to start.
        '''
