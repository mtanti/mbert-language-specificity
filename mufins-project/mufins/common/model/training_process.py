'''
Abstract base training process class.

Glossary:

- epoch: Visiting the entire training set during training.
- minibatch: A training set is divided into several minibatches of data, each of which is used
    to perform a parameter update in the model.
- validation check: After each epoch, the current model is used to evaluate the
    validation set in order to determine whether to stop training early or not.
    This process is called a 'validation check'.
    More specifically, a validation check event happens after the validation score is computed and
    a process-restoring checkpoint has been saved.
- patience: On each validation check, the score on the validation set is compared to the current
    best validation score. If this is not a new best score, the patience counter goes down by one,
    otherwise it resets back to maximum. Whether the patience reaches zero, training stops early.

Training checkpoint:

During training, on every validation check, a checkpoint is saved in order to be able to resume
training in case of interruption.
The checkpoint is saved in a SQLite database using the key '<model_id>/train', where <model_id> is
the unique name given to the model.
The value of the checkpoint is a JSON encoded string with the following information:

- best_score: The best validation score obtained.
- patience_left: The patience left.
- epoch_num: The last epoch number processed.
- ckpt_version: The checkpoint version (see atomic updates below).
- training_ready: Whether training finished.

Atomic updates:

The training process saves training state information such as the patience in the checkpoint file
and the model parameters in a pickle.
In order to avoid inconsistencies in case of abrupt interruption in between updating the checkpoint
and the pickle, an 'all-or-nothing' update (atomic) is made in the following way.
A checkpoint version number is used in the pickle file name.
The version number increments with every new checkpoint (validation check) and is saved in the
checkpoint's data.
When a new file is to be saved, the new file is saved alongside the previous one using different
version numbers.
The checkpoint is then updated, including the new version number which is part of the checkpoint.
The old file is then deleted.
This sequence of steps guarantees that either both the checkpoint and file are updated together or
the new checkpoint gets lost and the process will resume on the previous checkpoint instead.
This is because, given that the checkpoint is updated atomically (it's an SQLite database), if an
interruption happens after saving the file but before updating the checkpoint then the process will
resume on the old file because that is the checkpoint version found in the checkpoint.
On the other hand, if the interruption happens after updating the checkpoint but not before deleting
the old file then it will be of no consequence since only the new file will be used as that is the
version mentioned in the checkpoint.
The old file will be deleted when the process resumes later.
'''

import json
import os
import shutil
from abc import ABC, abstractmethod
from typing import Iterator, cast, Optional, Sequence, Mapping, TypeVar, Generic
import numpy as np
from mufins.common.checkpoint.checkpoint_manager import CheckpointManager
from mufins.common.time.time_utils import Timer, get_readable_duration
from mufins.common.log.log import Log
from mufins.common.model.model import Model


#########################################
M = TypeVar('M', bound=Model)

class TrainingProcess(ABC, Generic[M]):
    '''
    Provide a method for training a backend model using early stopping.
    '''
    #pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Empty constructor.
        '''

    #########################################
    def _on_epoch_start(
        self,
        epoch_num: int,
    ) -> None:
        '''
        Listener for when an epoch starts.

        :param epoch_num: The epoch number about to start.
        '''

    #########################################
    @abstractmethod
    def _train_epoch(
        self,
        epoch_num: int,
        minibatch_size: int,
        num_minibatches: Sequence[int],
        batch_size: int,
        model: M,
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

    #########################################
    @abstractmethod
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

    #########################################
    def _on_minibatch_end(
        self,
        epoch_num: int,
        minibatch_num: int,
    ) -> None:
        '''
        Listener for when a minibatch ends.

        :param epoch_num: The current epoch number.
        :param minibatch_num: The minibatch number just finished.
        '''

    #########################################
    @abstractmethod
    def _get_val_score(
        self,
        epoch_num: int,
    ) -> float:
        '''
        Get the validation score using the current model (score is to be maximised).

        :param epoch_num: The current epoch number.
        :return: The score.
        '''

    #########################################
    def _on_validation_check_end(
        self,
        epoch_num: int,
        new_best: bool,
        patience_left: Optional[int],
        curr_val_score: float,
        best_val_score: float,
        duration: float,
    ) -> None:
        '''
        Listener for when a validation check just finished.

        :param epoch_num: The current epoch number.
        :param new_best: Whether the validation score obtained was a new best score.
        :param patience_left: The amount of patience left after the validation check (can be zero,
            which means that training will end now).
        :param curr_val_score: The validation score obtained from the current validation check.
        :param best_val_score: The best validation score up to now.
        :param duration: The number of seconds elapsed since the last validation check or beginning
            of training if this was the first validation check.
        '''

    #########################################
    def _on_epoch_full_end(
        self,
        epoch_num: int,
        duration: float,
    ) -> None:
        '''
        Listener for when an epoch ends completely.

        :param epoch_num: The epoch number that just finished.
        :param duration: The duration of the epoch in seconds.
        '''

    #########################################
    def _on_epoch_early_end(
        self,
        epoch_num: int,
        duration: float,
    ) -> None:
        '''
        Listener for when an epoch ends early.

        :param epoch_num: The epoch number that just finished.
        :param duration: The duration of the epoch in seconds.
        '''

    #########################################
    def _on_epoch_max_epoch_end(
        self,
        epoch_num: int,
        duration: float,
    ) -> None:
        '''
        Listener for when an epoch ends due to the maximum number of epochs being reached.

        :param epoch_num: The epoch number that just finished.
        :param duration: The duration of the epoch in seconds.
        '''

    #########################################
    def _on_epoch_end(
        self,
        epoch_num: int,
        duration: float,
    ) -> None:
        '''
        Listener for when an epoch ends, regardless of how it ended.

        :param epoch_num: The epoch number that just finished.
        :param duration: The duration of the epoch in seconds.
        '''

    #########################################
    def __training_ended(
        # pylint: disable=no-self-use
        self,
        checkpoint_id: str,
        model: M,
        model_path: str,
        checkpoint_manager: CheckpointManager,
        ckpt_version: int,
    ) -> None:
        '''
        End of training process.

        :param checkpoint_id: A unique checkpoint key to use for checkpointing training progress.
        :param model: The backend model being trained.
        :param model_path: The path to the folder that will contain the model pickle files.
            One file called model.pkl will be used to store the best trained model
            (according to the validation set) whilst the other file called checkpoint.pkl
            will be used save a checkpoint model every validation check for resuming
            training in case of interruption.
        :param checkpoint_manager: The checkpoint manager.
        :param ckpt_version: The last checkpoint version number used.
        '''
        with checkpoint_manager.checkpoint(checkpoint_id) as handle:
            checkpoint_data = json.loads(cast(str, handle.get_value()))

        checkpoint_model_save_path = os.path.join(model_path, 'checkpoint_{}.pkl')
        best_model_save_path = os.path.join(model_path, 'model_{}.pkl')
        if not checkpoint_data['training_ready']:
            # Delete checkpoint pickle and rename model pickle atomically.
            shutil.copyfile(
                best_model_save_path.format(ckpt_version),
                best_model_save_path.replace('_{}', ''),
            )
            with checkpoint_manager.checkpoint(checkpoint_id) as handle:
                checkpoint_data['training_ready'] = True
                handle.set_value(json.dumps(checkpoint_data))
            os.remove(checkpoint_model_save_path.format(ckpt_version))
            os.remove(best_model_save_path.format(ckpt_version))

        model.load_params(best_model_save_path.replace('_{}', ''))

    #########################################
    def run(
        self,
        checkpoint_id: str,
        model: M,
        batch_size: int,
        minibatch_size: int,
        num_minibatches: Sequence[int],
        patience: Optional[int],
        max_epochs: Optional[int],
        model_path: str,
        checkpoint_manager: CheckpointManager,
        log: Optional[Log] = None,
    ) -> float:
        '''
        Run the process to train a model.

        :param checkpoint_id: A unique checkpoint key to use for checkpointing training progress.
        :param model: The backend model being trained.
        :param batch_size: The maximum amount of training items to process in one go.
        :param minibatch_size: The minibatch size to use whilst training.
        :param num_minibatches: The number of minibatches in each training set.
        :param patience: The number of less than best validation checks to allow
            before terminating training (if not None).
        :param max_epochs: Terminate training on the validation check that happens on or after this
            epoch number (if not None).
        :param model_path: The path to the folder that will contain the model pickle files.
            One file called model.pkl will be used to store the best trained model
            (according to the validation set) whilst the other file called checkpoint.pkl
            will be used save a checkpoint model every validation check for resuming
            training in case of interruption.
        :param checkpoint_manager: The checkpoint manager.
        :param log: The log to monitor training progress.
            If None then no progress will be monitored.
        :return: The best validation score obtained by the end of training.
        '''
        checkpoint_model_save_path = os.path.join(model_path, 'checkpoint_{}.pkl')
        best_model_save_path = os.path.join(model_path, 'model_{}.pkl')

        checkpoint_found = True
        with checkpoint_manager.checkpoint(checkpoint_id) as handle:
            if not handle.was_found_ready():
                with Timer() as epoch_timer:
                    best_score = self._get_val_score(epoch_num=0)

                patience_left = patience
                start_epoch_num = 1
                ckpt_version = 0
                model.save_state(checkpoint_model_save_path.format(ckpt_version))
                model.save_params(best_model_save_path.format(ckpt_version))
                handle.set_value(json.dumps({
                    'best_score': best_score,
                    'patience_left': patience_left,
                    'epoch_num': start_epoch_num,
                    'ckpt_version': ckpt_version,
                    'training_ready': False,
                }))
                checkpoint_found = False

                self._on_validation_check_end(
                    epoch_num=0, new_best=True, patience_left=patience,
                    curr_val_score=best_score, best_val_score=best_score,
                    duration=epoch_timer.get_duration(),
                )

        if checkpoint_found:
            checkpoint_data = json.loads(cast(str, handle.get_value()))
            best_score = checkpoint_data['best_score']
            patience_left = checkpoint_data['patience_left']
            start_epoch_num = checkpoint_data['epoch_num']
            ckpt_version = checkpoint_data['ckpt_version']
            assert isinstance(best_score, float), best_score
            assert isinstance(patience_left, (int, type(None))), patience_left
            assert isinstance(start_epoch_num, int), start_epoch_num
            assert isinstance(ckpt_version, int), ckpt_version
            if os.path.isfile(checkpoint_model_save_path.format(ckpt_version - 1)):
                os.remove(checkpoint_model_save_path.format(ckpt_version - 1))
            if os.path.isfile(best_model_save_path.format(ckpt_version - 1)):
                os.remove(best_model_save_path.format(ckpt_version - 1))

            if checkpoint_data['training_ready']:
                if log is not None:
                    log.log_message('Loading pre-trained model.')
                self.__training_ended(
                    checkpoint_id=checkpoint_id,
                    model=model,
                    model_path=model_path,
                    checkpoint_manager=checkpoint_manager,
                    ckpt_version=ckpt_version,
                )
                return best_score
            model.load_state(checkpoint_model_save_path.format(ckpt_version))

            if log is not None:
                log.log_message(
                    'Continuing training for this model from where it was left off.'
                    ' Starting on epoch {}.'.format(
                        start_epoch_num
                    )
                )

        epoch_num = start_epoch_num - 1
        while True:
            epoch_num += 1

            self._on_epoch_start(epoch_num)
            with Timer() as epoch_timer:
                val_score = self._train_epoch(
                    epoch_num=epoch_num,
                    minibatch_size=minibatch_size,
                    num_minibatches=num_minibatches,
                    batch_size=batch_size,
                    model=model,
                    log=log,
                )

                # Update stats
                if val_score > best_score:
                    patience_left = patience
                    best_score = val_score
                    new_best = True
                else:
                    if patience_left is not None:
                        patience_left -= 1
                    new_best = False

                # Save checkpoint atomically
                ckpt_version += 1
                model.save_state(checkpoint_model_save_path.format(ckpt_version))
                if new_best:
                    model.save_params(best_model_save_path.format(ckpt_version))
                else:
                    shutil.copyfile(
                        best_model_save_path.format(ckpt_version - 1),
                        best_model_save_path.format(ckpt_version),
                    )
                with checkpoint_manager.checkpoint(checkpoint_id) as handle:
                    handle.set_value(json.dumps({
                        'best_score': best_score,
                        'patience_left': patience_left,
                        'epoch_num': epoch_num + 1,
                        'ckpt_version': ckpt_version,
                        'training_ready': False,
                    }))
                os.remove(best_model_save_path.format(ckpt_version - 1))
                os.remove(checkpoint_model_save_path.format(ckpt_version - 1))

            epoch_duration = epoch_timer.get_duration()

            self._on_validation_check_end(
                epoch_num, new_best, patience_left,
                val_score, best_score,
                epoch_duration,
            )

            # Early stopping.
            if patience_left == 0:
                if log is not None:
                    log.log_message('Finished epoch {} early: {}'.format(
                        epoch_num, get_readable_duration(epoch_duration),
                    ))
                self._on_epoch_early_end(
                    epoch_num, epoch_duration,
                )
                self._on_epoch_end(
                    epoch_num, epoch_duration,
                )

                self.__training_ended(
                    checkpoint_id=checkpoint_id,
                    model=model,
                    model_path=model_path,
                    checkpoint_manager=checkpoint_manager,
                    ckpt_version=ckpt_version,
                )
                return best_score

            # Maximum epoch reached.
            if max_epochs is not None and epoch_num >= max_epochs:
                if log is not None:
                    log.log_message('Max epochs reached on epoch {}: {}'.format(
                        epoch_num, get_readable_duration(epoch_duration),
                    ))
                self._on_epoch_max_epoch_end(
                    epoch_num, epoch_duration,
                )
                self._on_epoch_end(
                    epoch_num, epoch_duration,
                )

                self.__training_ended(
                    checkpoint_id=checkpoint_id,
                    model=model,
                    model_path=model_path,
                    checkpoint_manager=checkpoint_manager,
                    ckpt_version=ckpt_version,
                )
                return best_score

            if log is not None:
                log.log_message('Finished epoch {}: {}'.format(
                    epoch_num, get_readable_duration(epoch_duration),
                ))
            self._on_epoch_full_end(epoch_num, epoch_duration)
            self._on_epoch_end(epoch_num, epoch_duration)
