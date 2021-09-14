'''
Training process for the model.
'''

import os
from typing import Iterator, Optional, Mapping, Sequence
import numpy as np
from mufins.common.log.log import Log
from mufins.common.file.csv_file import CsvFile
from mufins.common.dataset.dataset import Dataset
from mufins.common.random.random_number_generator import RandomNumberGenerator
from mufins.common.model.training_process_adversarial import TrainingProcessAdversarial
from mufins.dataprocs.xnli.data_row import XNLIDataRow
from mufins.dataprocs.xnli.data_spec import XNLIDataSpec
from mufins.dataprocs.wikipedia.data_row import WikipediaDataRow
from mufins.dataprocs.wikipedia.data_spec import WikipediaDataSpec
from mufins.experiments.lang_ent_max_cls.evaluate import (
    eval_label, eval_lang,
    LabelEncoderType, LangEncoderType, LabelPredictorType, LangPredictorType
)


#########################################
class ModelTrainingProcess(TrainingProcessAdversarial):
    '''
    The training process specification for this model.
    '''
    # pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        rng: RandomNumberGenerator,
        batch_size: int,
        label_encoder: LabelEncoderType,
        lang_encoder: LangEncoderType,
        label_predictor: LabelPredictorType,
        lang_predictor: LangPredictorType,
        dset_label_train: Dataset[XNLIDataRow],
        dset_lang_train: Dataset[WikipediaDataRow],
        dset_label_val: Dataset[XNLIDataRow],
        dset_lang_val: Dataset[WikipediaDataRow],
        label_spec: XNLIDataSpec,
        lang_spec: WikipediaDataSpec,
        training_main_module: bool,
        hyperparameter_search_mode: bool,
        train_history_path: str,
        log: Log,
    ) -> None:
        '''
        Constructor.

        :param rng: The random number generator to use.
        :param batch_size: The maximum number of data items to process at once.
        :param label_encoder: A function that encodes label texts into vectors.
        :param lang_encoder: A function that encodes language texts into vectors.
        :param label_predictor: A function that predicts labels.
        :param lang_predictor: A function that predicts languages.
        :param dset_label_train: Label training set.
        :param dset_lang_train: Language training set.
        :param dset_label_val: Label validation set.
        :param dset_lang_val: Language validation set.
        :param label_spec: The label data specification.
        :param lang_spec: The language data specification.
        :param training_main_module: Whether the main module is being trained, otherwise the
            language module will be trained.
        :param hyperparameter_search_mode: Whether to enter into hyperparameter search mode where
            minimal output and evaluation is produced.
        :param train_history_path: The path to the folder that will contain the training history.
        :param log: The log.
        '''
        super().__init__(training_main_module)

        self.rng: RandomNumberGenerator = rng
        self.batch_size: int = batch_size
        self.label_encoder: LabelEncoderType = label_encoder
        self.lang_encoder: LangEncoderType = lang_encoder
        self.label_predictor: LabelPredictorType = label_predictor
        self.lang_predictor: LangPredictorType = lang_predictor
        self.dset_label_train: Dataset[XNLIDataRow] = dset_label_train
        self.dset_lang_train: Dataset[WikipediaDataRow] = dset_lang_train
        self.dset_label_val: Dataset[XNLIDataRow] = dset_label_val
        self.dset_lang_val: Dataset[WikipediaDataRow] = dset_lang_val
        self.label_spec: XNLIDataSpec = label_spec
        self.lang_spec: WikipediaDataSpec = lang_spec
        self.hyperparameter_search_mode: bool = hyperparameter_search_mode
        self.train_history_path: str = train_history_path
        self.log: Log = log

        self.training_main_module: bool = training_main_module

        self.post_disc_score_lang_val: float = 0.0
        self.lang_val_min_prob_entropy: float = 0.0
        self.lang_val_mean_prob_entropy: float = 0.0
        self.lang_val_max_prob_entropy: float = 0.0
        self.score_label_label_val: float = 0.0
        self.score_label_lang_val: float = 0.0
        self.score_lang_val: float = 0.0

        self.train_history_file = CsvFile(
            os.path.join(train_history_path, 'train_history.csv'),
        )
        if not self.hyperparameter_search_mode:
            self.train_history_file.init([
                'phase',
                'epoch',
                'new_best',
                'patience_left',
                'post_disc_lang_val_macro_f1_score',
                'label_label_val_macro_f1_score',
                'label_lang_val_macro_f1_score',
                'lang_val_macro_f1_score',
                'lang_val_min_entropy',
                'lang_val_mean_entropy',
                'lang_val_max_entropy',
            ])

    #########################################
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
        assert self.training_main_module
        capped_size = min(self.dset_label_train.size, self.dset_lang_train.size)
        return zip(
            self.dset_lang_train.get_stochastic_batches(
                minibatch_size, self.rng, capped_size=capped_size,
            )
        )

    #########################################
    def _on_discriminator_trained(
        self,
        epoch_num: int,
    ) -> None:
        '''
        Listener for when the disciminator has been trained, before training the rest of the model.

        :param epoch_num: The epoch number about to start.
        '''
        assert self.training_main_module
        if not self.hyperparameter_search_mode:
            (
                lang_score_f1_macro,
                _,
                _,
                _,
            ) = eval_lang(
                lang_spec=self.lang_spec,
                dset_lang=self.dset_lang_val,
                lang_predictor=self.lang_predictor,
                batch_size=self.batch_size,
                log=self.log,
            )
            self.post_disc_score_lang_val = lang_score_f1_macro

    #########################################
    def _get_minibatches(
        # pylint: disable=unused-argument
        self,
        epoch_num: int,
        minibatch_size: int
    ) -> Iterator[Sequence[Mapping[str, np.ndarray]]]:
        '''
        Get an iterator over all the batches in the training set.

        These batches will be passed to `batch_fit` in the backend model.

        :param epoch_num: The epoch number about to start.
        :param minibatch_size: The number of items in a minibatch.
        :return: The iterator of batches.
        '''
        if self.training_main_module:
            return zip(
                self.dset_label_train.get_stochastic_batches(
                    minibatch_size, self.rng,
                ),
                self.dset_lang_train.get_stochastic_batches(
                    minibatch_size, self.rng,
                    capped_size=self.dset_label_train.size,
                ),
            )
        return zip(
            self.dset_lang_train.get_stochastic_batches(
                minibatch_size, self.rng,
            )
        )

    #########################################
    def _get_val_score(
        # pylint: disable=unused-argument
        self,
        epoch_num: int,
    ) -> float:
        '''
        Get the validation score using the current model (score is to be maximised).

        :param epoch_num: The current epoch number.
        :return: The score.
        '''
        if self.training_main_module:
            (
                label_label_score_f1_macro,
                _,
                _,
                label_lang_score_f1_macro,
            ) = eval_label(
                label_spec=self.label_spec,
                lang_spec=self.lang_spec,
                dset_label=self.dset_label_val,
                label_predictor=self.label_predictor,
                lang_predictor=self.lang_predictor,
                hyperparameter_search_mode=self.hyperparameter_search_mode,
                batch_size=self.batch_size,
                log=self.log,
            )
            self.score_label_label_val = label_label_score_f1_macro
            self.score_label_lang_val = label_lang_score_f1_macro

        if not self.hyperparameter_search_mode:
            (
                lang_score_f1_macro,
                min_prob_entropy,
                mean_prob_entropy,
                max_prob_entropy,
            ) = eval_lang(
                lang_spec=self.lang_spec,
                dset_lang=self.dset_lang_val,
                lang_predictor=self.lang_predictor,
                batch_size=self.batch_size,
                log=self.log,
            )
            self.score_lang_val = lang_score_f1_macro
            self.lang_val_min_prob_entropy = min_prob_entropy
            self.lang_val_mean_prob_entropy = mean_prob_entropy
            self.lang_val_max_prob_entropy = max_prob_entropy

        if self.training_main_module:
            return self.score_label_label_val
        return self.score_lang_val

    #########################################
    def _on_validation_check_end(
        # pylint: disable=unused-argument
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
        if not self.hyperparameter_search_mode:
            if self.training_main_module:
                self.train_history_file.append([
                    'main',
                    str(epoch_num),
                    'yes' if new_best else 'no',
                    str(patience_left),
                    '{:.10f}'.format(self.post_disc_score_lang_val),
                    '{:.10f}'.format(self.score_label_label_val),
                    '{:.10f}'.format(self.score_label_lang_val),
                    '{:.10f}'.format(self.score_lang_val),
                    '{:.10f}'.format(self.lang_val_min_prob_entropy),
                    '{:.10f}'.format(self.lang_val_mean_prob_entropy),
                    '{:.10f}'.format(self.lang_val_max_prob_entropy),
                ])
            else:
                self.train_history_file.append([
                    'lang',
                    str(epoch_num),
                    'yes' if new_best else 'no',
                    str(patience_left),
                    '',
                    '',
                    '',
                    '{:.10f}'.format(self.score_lang_val),
                    '{:.10f}'.format(self.lang_val_min_prob_entropy),
                    '{:.10f}'.format(self.lang_val_mean_prob_entropy),
                    '{:.10f}'.format(self.lang_val_max_prob_entropy),
                ])
