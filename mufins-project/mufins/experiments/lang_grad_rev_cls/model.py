'''
Neural network model for the experiment.
'''

import os
import json
import itertools
from typing import Mapping, Optional, Tuple, Dict, Sequence, List, Union, Iterator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import transformers
import numpy as np
from mufins.common.random.random_number_generator import RandomNumberGenerator
from mufins.common.dataset.dataset import (
    Dataset, get_batch_size, get_subbatches, concatenate_subbatches
)
from mufins.common.encoder.encoder import Encoder
from mufins.common.encoder.encoder_factory import encoder_factory
from mufins.common.checkpoint.checkpoint_manager import CheckpointManager
from mufins.common.model.model_standard import ModelStandard
from mufins.common.model.grad_rev_layer import grad_rev_layer
from mufins.common.log.log import Log
from mufins.experiments.lang_grad_rev_cls.hyperparameters import Hyperparameters
from mufins.experiments.lang_grad_rev_cls.training_process import ModelTrainingProcess
from mufins.dataprocs.xnli.data_row import XNLIDataRow
from mufins.dataprocs.wikipedia.data_row import WikipediaDataRow
from mufins.dataprocs.xnli.data_spec import XNLIDataSpec
from mufins.dataprocs.wikipedia.data_spec import WikipediaDataSpec


transformers.logging.set_verbosity_error()


#########################################
class Net(nn.Module):
    '''
    The neural network definition.
    '''

    #########################################
    def __init__(
        self,
        hyperparams: Hyperparameters,
        label_spec: XNLIDataSpec,
        lang_spec: WikipediaDataSpec,
        rng: RandomNumberGenerator,
    ) -> None:
        '''
        Constructor.

        :param hyperparams: The hyperparameters of the model.
        :param label_spec: The label data specifier that this model works with.
        :param lang_spec: The language data specifier that this model works with.
        :param rng: The random number generator to use.
        '''
        super().__init__()

        self.grl_lambda: float = hyperparams.grl_lambda

        self.encoder: Encoder = encoder_factory(
            hyperparams.encoder_name, hyperparams.layer_index
        )

        self.dropout_layer: nn.Dropout = nn.Dropout(hyperparams.dropout_rate)

        self.label_logits: nn.Linear = nn.Linear(768, len(label_spec.label_names))
        self.label_logits.weight.data = torch.tensor(
            rng.array_normal(
                0.0,
                hyperparams.init_stddev,
                self.label_logits.weight.shape,
            ),
            dtype=torch.float32
        )
        self.label_logits.bias.data = torch.zeros_like(
            self.label_logits.bias
        )

        self.lang_logits: nn.Linear = nn.Linear(768, len(lang_spec.lang_names))
        self.initialise_lang_module(hyperparams, rng)

    #########################################
    def initialise_lang_module(
        self,
        hyperparams: Hyperparameters,
        rng: RandomNumberGenerator,
    ) -> None:
        '''
        Constructor.

        :param hyperparams: The hyperparameters of the model.
        :param rng: The random number generator to use.
        '''
        self.lang_logits.weight.data = torch.tensor(
            rng.array_normal(
                0.0,
                hyperparams.init_stddev,
                self.lang_logits.weight.shape,
            ),
            dtype=torch.float32
        )
        self.lang_logits.bias.data = torch.zeros_like(
            self.lang_logits.bias
        )

    #########################################
    def forward(
        self,
        label_token_indexes: Optional[torch.Tensor],
        label_token_mask: Optional[torch.Tensor],
        lang_token_indexes: Optional[torch.Tensor],
        lang_token_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        '''
        Forward pass.

        :param label_token_indexes: Integer array of token indexes to be
            used for predicting the labels.
        :param label_token_mask: Boolean array of token masks to be
            used for predicting the labels.
        :param lang_token_indexes: Integer array of token indexes to be
            used for predicting the language.
        :param lang_token_mask: Boolean array of token masks to be
            used for predicting the language.
        :return: The logits of the label classification and language
            classification as a tuple. If label_token_indexes is None
            then the first item will be None. If lang_token_indexes is
            None then the second item will be None.
        '''
        if label_token_indexes is not None and label_token_mask is not None:
            text_vecs = self.encoder.encode_texts(label_token_indexes, label_token_mask)
            text_vecs = self.dropout_layer(text_vecs)

            label_logits = self.label_logits(text_vecs)
        else:
            label_logits = None

        if lang_token_indexes is not None and lang_token_mask is not None:
            text_vecs = self.encoder.encode_texts(lang_token_indexes, lang_token_mask)
            text_vecs = self.dropout_layer(text_vecs)
            if self.training:
                text_vecs = grad_rev_layer(text_vecs, self.grl_lambda)
            lang_logits = self.lang_logits(text_vecs)
        else:
            lang_logits = None

        return (label_logits, lang_logits)


#########################################
class LangGradRevClsModel(ModelStandard):
    '''
    The model class.
    '''

    #########################################
    def __init__(
        self,
        label_spec: XNLIDataSpec,
        lang_spec: WikipediaDataSpec,
        device_name: str,
        hyperparams: Hyperparameters,
        rng: RandomNumberGenerator,
    ) -> None:
        '''
        Constructor.

        :param label_spec: The label data specifier that this model works with.
        :param lang_spec: The language data specifier that this model works with.
        :param device_name: The GPU device name to use such as 'cuda:0'. Use 'cpu'
            if no GPU is available.
        :param hyperparams: The hyperparameters of the model.
        :param rng: The random number generator to use.
        '''
        self.label_spec: XNLIDataSpec = label_spec
        self.lang_spec: WikipediaDataSpec = lang_spec
        self.hyperparams: Hyperparameters = hyperparams
        self.rng: RandomNumberGenerator = rng.get_child()

        device = ModelStandard.get_device(device_name)
        net = Net(
            hyperparams, label_spec, lang_spec, rng.get_child(),
        )
        opt_main = optim.Adam(
            (
                [
                    {
                        'params': net.encoder.get_parameters(
                            with_embeddings=not hyperparams.freeze_embeddings
                        ),
                        'lr': self.hyperparams.encoder_learning_rate,
                    }
                ] if self.hyperparams.encoder_learning_rate is not None else []
            ) + [
                {
                    'params': itertools.chain(
                        net.label_logits.parameters(),
                        net.lang_logits.parameters(),
                    ),
                    'lr': self.hyperparams.postencoder_learning_rate,
                },
            ]
        )
        opt_lang = optim.Adam([
            {
                'params': net.lang_logits.parameters(),
                'lr': self.hyperparams.postencoder_learning_rate,
            },
        ])

        super().__init__(net, [opt_main, opt_lang], device)

        self.net: Net = net
        self.encoder: Encoder = net.encoder

    #########################################
    def save_model(
        self,
        model_path: str,
        params_path: Optional[str] = None,
    ) -> None:
        '''
        Save the whole model definition.

        :param model_path: The path to the model definition (.json).
        :param params_path: The path to the model parameters (.pkl), if they should be saved.
        '''
        obj = {
            'label_spec': json.loads(self.label_spec.to_json()),
            'lang_spec': json.loads(self.lang_spec.to_json()),
            'hyperparams': json.loads(self.hyperparams.to_json()),
        }
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=1)
        if params_path is not None:
            self.save_params(params_path)

    #########################################
    @staticmethod
    def load_model(
        model_path: str,
        device_name: str,
        rng: RandomNumberGenerator,
        params_path: Optional[str] = None,
    ) -> 'LangGradRevClsModel':
        '''
        Load a model from definition.

        :param model_path: The path to the model definition (.json).
        :param device_name: The GPU device name to use such as 'cuda:0'. Use 'cpu'
            if no GPU is available.
        :param rng: The random number generator to use.
        :param params_path: The path to the model parameters (.pkl), if they should be loaded.
        :return: The loaded model.
        '''
        with open(model_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        model = LangGradRevClsModel(
            label_spec=XNLIDataSpec.from_json(json.dumps(obj['label_spec'])),
            lang_spec=WikipediaDataSpec.from_json(json.dumps(obj['lang_spec'])),
            device_name=device_name,
            hyperparams=Hyperparameters.from_json(json.dumps(obj['hyperparams'])),
            rng=rng,
        )
        if params_path is not None:
            model.load_params(params_path)
        return model

    #########################################
    def __batch_prepare_inputs(
        self,
        label_batch: Optional[Mapping[str, np.ndarray]],
        lang_batch: Optional[Mapping[str, np.ndarray]],
    ) -> Tuple[
        Optional[Mapping[str, torch.Tensor]],
        Optional[Mapping[str, torch.Tensor]],
    ]:
        '''
        Transform a batch of data into tensor inputs.

        :param label_batch: A dictionary that maps field names to numpy arrays
            that are to be processed as a single batch for label classification.
            Use the data specifier `label_spec` to obtain dictionary.
        :param lang_batch: A dictionary that maps field names to numpy arrays
            that are to be processed as a single batch for language classification.
            Use the data specifier `lang_spec` to obtain dictionary.
        :return: A pair consisting of two mappings, one for the label part of the model and one for
            the language part. Each mapping contains the following input items:

            - tokens: A PyTorch tensor with the token indexes.
            - label: A PyTorch tensor with the label indexes.
            - token_mask: A PyTorch tensor with the mask used for the sentence.

            The mapping will be replaced with None if the corresponding batch is None.
        '''
        result: Dict[str, Optional[Dict[str, torch.Tensor]]] = {
            'label': None,
            'lang': None,
        }

        for (key, batch) in [
            ('label', label_batch),
            ('lang', lang_batch),
        ]:
            if batch is None:
                result[key] = None
                continue

            token_indexes = batch['tokens']
            if key == 'label':
                label_indexes = batch['label']
            else:
                label_indexes = batch['lang']
            num_tokens = batch['num_tokens']
            curr_max_len = np.max(num_tokens).tolist()
            curr_batch_size = token_indexes.shape[0]

            token_indexes_tensor = torch.tensor(
                token_indexes[:, :curr_max_len].astype(np.int64),
                dtype=torch.int64,
                device=self.device,
            )

            token_mask = np.zeros([curr_batch_size, curr_max_len], np.float32)
            for j in range(curr_batch_size):
                token_mask[j, :num_tokens[j]] = 1.0

            token_mask_tensor = torch.tensor(
                token_mask,
                dtype=torch.float32,
                device=self.device,
            )

            label_indexes_tensor = torch.tensor(
                label_indexes.astype(np.int64),
                dtype=torch.int64,
                device=self.device,
            )

            result[key] = {
                'tokens': token_indexes_tensor,
                'token_mask': token_mask_tensor,
                'label': label_indexes_tensor,
            }

        return (result['label'], result['lang'])

    #########################################
    def __batch_encoder(
        self,
        prepared_batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        '''
        Get the encoding vector of the text before being passed to the classifier.

        :param prepared_batch: The needed item returned by `__batch_prepare_inputs`.
        :return: The encoding vector of the text.
        '''
        with torch.no_grad():
            return self.encoder.encode_texts(
                prepared_batch['tokens'], prepared_batch['token_mask']
            )

    #########################################
    def __batch_logits(
        self,
        prepared_label_batch: Optional[Mapping[str, torch.Tensor]],
        prepared_lang_batch: Optional[Mapping[str, torch.Tensor]],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor]
    ]:
        '''
        Output the pre-classification logits for each text in a batch of data.

        :param prepared_label_batch: The first item returned by `__batch_prepare_inputs`.
        :param prepared_lang_batch: The second item returned by `__batch_prepare_inputs`.
        :return: A pair consisting of the logits of the label classifier and the language
            classifier.
            The logits will be replaced with None if the corresponding input is None.
        '''
        (label_logits, lang_logits) = self.net(
            prepared_label_batch['tokens'] if prepared_label_batch is not None else None,
            prepared_label_batch['token_mask'] if prepared_label_batch is not None else None,
            prepared_lang_batch['tokens'] if prepared_lang_batch is not None else None,
            prepared_lang_batch['token_mask'] if prepared_lang_batch is not None else None,
        )
        return (label_logits, lang_logits)

    #########################################
    def __batch_logprobs(
        self,
        prepared_label_batch: Optional[Mapping[str, torch.Tensor]],
        prepared_lang_batch: Optional[Mapping[str, torch.Tensor]],
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        '''
        Predict the log_e probability of each label for each text in a batch.

        :param prepared_label_batch: The first item returned by `__batch_prepare_inputs`.
        :param prepared_lang_batch: The second item returned by `__batch_prepare_inputs`.
        :return: A pair consisting of the log_e probabilities of the label classifier and the
            language classifier.
            The logits will be replaced with None if the corresponding input is None.
        '''
        with torch.no_grad():
            (label_logits, lang_logits) = self.__batch_logits(
                prepared_label_batch, prepared_lang_batch
            )

            if label_logits is None:
                label_logprobs = None
            else:
                label_logprobs = F.log_softmax(label_logits, dim=1).detach().cpu().numpy()

            if lang_logits is None:
                lang_logprobs = None
            else:
                lang_logprobs = F.log_softmax(lang_logits, dim=1).detach().cpu().numpy()

        return (label_logprobs, lang_logprobs)

    #########################################
    def get_label_encodings(
        self,
        dset_label: Dataset[XNLIDataRow],
        batch_size: int,
        log: Optional[Log] = None,
        progress_label: Optional[str] = None,
        sampler: Optional[RandomNumberGenerator] = None,
        sample_size_per_unit: Optional[int] = None,
    ) -> Tuple[
        Sequence[Sequence[float]],
        Sequence[str],
        Sequence[str]
    ]:
        '''
        For each text in the dataset, output the encoding vector of each text, the
        name of the true language of each text, and the name of the true label.

        :param dset_label: The dataset of sentences to make predictions about.
        :param batch_size: The maximum amount of sentences to process in one go.
        :param log: The log to pass progress information to.
        :param progress_label: The label to use in the progress information, if a log is provided.
        :param sampler: The random number generator to use to sample texts to encode.
            If None then all texts are encoded.
        :param sample_size_per_unit: The maximum number of each language-label combination to
            sample.
            Must be None if and only if sampler is None.
        :return: A triple consisting of the following:

            - A list with an item for each text consisting of a list of feature elements.
            - A list of the names of the true language for each text.
            - A list of the names of the true label for each text.
        '''
        if dset_label.spec != self.label_spec:
            raise ValueError('Dataset is incompatible with model.')
        if (sampler is None) != (sample_size_per_unit is None):
            raise ValueError('Sampler is None whilst sample size per unit is not, or vice versa.')

        batches: Iterator[Mapping[str, np.ndarray]] = dset_label.get_batches(batch_size)
        data_size = dset_label.size
        if sampler is not None:
            def get_sample() -> Tuple[Iterator[Mapping[str, np.ndarray]], int]:
                assert sampler is not None
                assert sample_size_per_unit is not None
                indexes = list(dset_label.get_data(
                    batch_size=batch_size,
                    value_mapper=lambda row_index, row:(row_index, row['lang'], row['label'])
                ))
                sampler.shuffle(indexes)
                freqs = {
                    (lang, label): 0
                    for lang in self.label_spec.lang_names
                    for label in self.label_spec.label_names
                }
                sample: List[Mapping[str, np.ndarray]] = list()
                for (row_index, lang_index, label_index) in indexes:
                    lang = self.label_spec.lang_names[lang_index.tolist()]
                    label = self.label_spec.label_names[label_index.tolist()]
                    if freqs[(lang, label)] < sample_size_per_unit:
                        sample.append(dset_label.get_row(row_index))
                        freqs[(lang, label)] += 1
                batches = get_subbatches(concatenate_subbatches(sample), batch_size)
                data_size = len(sample)
                return (batches, data_size)
            (batches, data_size) = get_sample()

        full_encodings: List[List[float]] = list()
        full_langs: List[str] = list()
        full_labels: List[str] = list()

        sent_index = 0
        if log is not None:
            log.progress_start(0, data_size, progress_label)
        for batch in batches:
            (tensors, _) = self.__batch_prepare_inputs(batch, None)
            assert tensors is not None
            encodings = self.__batch_encoder(tensors)

            for row_idx in range(encodings.shape[0]):
                full_encodings.append(
                    encodings[row_idx, :].tolist()
                )
                full_langs.append(self.label_spec.lang_names[
                    batch['lang'][row_idx].tolist()
                ])
                full_labels.append(self.label_spec.label_names[
                    batch['label'][row_idx].tolist()
                ])
                sent_index += 1
            if log is not None:
                log.progress_update(sent_index)
        if log is not None:
            log.progress_end()

        return (full_encodings, full_langs, full_labels)

    #########################################
    def get_lang_encodings(
        self,
        dset_lang: Dataset[WikipediaDataRow],
        batch_size: int,
        log: Optional[Log] = None,
        progress_label: Optional[str] = None,
        sampler: Optional[RandomNumberGenerator] = None,
        sample_size_per_unit: Optional[int] = None,
    ) -> Tuple[
        Sequence[Sequence[float]],
        Sequence[str]
    ]:
        '''
        For each text in the dataset, output the encoding vector of each text and the
        name of the true language of each text.

        :param dset_lang: The dataset of sentences to make predictions about.
        :param batch_size: The maximum amount of sentences to process in one go.
        :param log: The log to pass progress information to.
        :param progress_label: The label to use in the progress information, if a log is provided.
        :param sampler: The random number generator to use to sample texts to encode.
            If None then all texts are encoded.
        :param sample_size_per_unit: The maximum number of each language to sample.
            Must be None if and only if sampler is None.
        :return: A triple consisting of the following:

            - A list with an item for each text consisting of a list of feature elements.
            - A list of the names of the true language for each text.
        '''
        if dset_lang.spec != self.lang_spec:
            raise ValueError('Dataset is incompatible with model.')
        if (sampler is None) != (sample_size_per_unit is None):
            raise ValueError('Sampler is None whilst sample size per unit is not, or vice versa.')

        batches: Iterator[Mapping[str, np.ndarray]] = dset_lang.get_batches(batch_size)
        data_size = dset_lang.size
        if sampler is not None:
            def get_sample() -> Tuple[Iterator[Mapping[str, np.ndarray]], int]:
                assert sampler is not None
                assert sample_size_per_unit is not None
                indexes = list(dset_lang.get_data(
                    batch_size=batch_size,
                    value_mapper=lambda row_index, row:(row_index, row['lang'])
                ))
                sampler.shuffle(indexes)
                freqs = {
                    lang: 0
                    for lang in self.lang_spec.lang_names
                }
                sample: List[Mapping[str, np.ndarray]] = list()
                for (row_index, lang_index) in indexes:
                    lang = self.lang_spec.lang_names[lang_index.tolist()]
                    if freqs[lang] < sample_size_per_unit:
                        sample.append(dset_lang.get_row(row_index))
                        freqs[lang] += 1
                batches = get_subbatches(concatenate_subbatches(sample), batch_size)
                data_size = len(sample)
                return (batches, data_size)
            (batches, data_size) = get_sample()

        full_encodings: List[List[float]] = list()
        full_langs: List[str] = list()

        sent_index = 0
        if log is not None:
            log.progress_start(0, data_size, progress_label)
        for batch in batches:
            (_, tensors) = self.__batch_prepare_inputs(None, batch)
            assert tensors is not None
            encodings = self.__batch_encoder(tensors)

            for row_idx in range(encodings.shape[0]):
                full_encodings.append(
                    encodings[row_idx, :].tolist()
                )
                full_langs.append(self.lang_spec.lang_names[
                    batch['lang'][row_idx].tolist()
                ])
                sent_index += 1
            if log is not None:
                log.progress_update(sent_index)
        if log is not None:
            log.progress_end()

        return (full_encodings, full_langs)

    #########################################
    def get_label_predictions(
        self,
        dset_label: Dataset[XNLIDataRow],
        batch_size: int,
        log: Optional[Log] = None,
        progress_label: Optional[str] = None,
    ) -> Tuple[
        Sequence[Sequence[float]],
        Sequence[str],
        Sequence[str]
    ]:
        '''
        For each text in the dataset, predict the log_e probabilities of each label, the
        name of the most likely label, and the name of the true label (from the dataset).

        :param dset_label: The dataset of sentences to make predictions about.
        :param batch_size: The maximum amount of sentences to process in one go.
        :param log: The log to pass progress information to.
        :param progress_label: The label to use in the progress information, if a log is provided.
        :return: A triple consisting of the following: A list with an item for each text
            consisting of a list of the log_e probabilities of each label, a list of the names
            of the most likely label for each text, a list of the names of the true label
            for each text.
        '''
        if dset_label.spec != self.label_spec:
            raise ValueError('Dataset is incompatible with model.')

        full_probs = list()
        full_preds = list()
        full_trues = list()

        sent_index = 0
        if log is not None:
            log.progress_start(0, dset_label.size, progress_label)
        for batch in dset_label.get_batches(batch_size):
            (logprobs, _) = self.__batch_logprobs(*self.__batch_prepare_inputs(batch, None))
            assert logprobs is not None
            predicted_labels = logprobs.argmax(axis=1)

            for row_idx in range(predicted_labels.shape[0]):
                full_probs.append(
                    logprobs[row_idx, :].tolist()
                )
                full_preds.append(self.label_spec.label_names[
                    predicted_labels[row_idx].tolist()
                ])
                full_trues.append(self.label_spec.label_names[
                    batch['label'][row_idx].tolist()
                ])
                sent_index += 1
            if log is not None:
                log.progress_update(sent_index)
        if log is not None:
            log.progress_end()

        return (full_probs, full_preds, full_trues)

    #########################################
    def get_lang_predictions(
        self,
        dset: Union[Dataset[WikipediaDataRow], Dataset[XNLIDataRow]],
        batch_size: int,
        log: Optional[Log] = None,
        progress_label: Optional[str] = None,
    ) -> Tuple[
        Sequence[Sequence[float]],
        Sequence[str],
        Sequence[str],
    ]:
        '''
        For each text in the dataset, predict the log_e probabilities of each language, the
        name of the most likely language, and the name of the true language (from the dataset).

        :param dset: The dataset of sentences to make predictions about.
        :param batch_size: The maximum amount of sentences to process in one go.
        :param log: The log to pass progress information to.
        :param progress_label: The label to use in the progress information, if a log is provided.
        :return: A triple consisting of the following: A list with an item for each text
            consisting of a list of the log_e probabilities of each language, a list of the names
            of the most likely language for each text, a list of the names of the true language
            for each text.
        '''
        if dset.spec not in [self.lang_spec, self.label_spec]:
            raise ValueError('Dataset is incompatible with model.')

        if dset.spec == self.lang_spec:
            dset_lang_names = self.lang_spec.lang_names
        else:
            dset_lang_names = self.label_spec.lang_names

        full_probs = list()
        full_preds = list()
        full_trues = list()

        sent_index = 0
        if log is not None:
            log.progress_start(0, dset.size, progress_label)
        for batch in dset.get_batches(batch_size):
            (_, logprobs) = self.__batch_logprobs(*self.__batch_prepare_inputs(None, batch))
            assert logprobs is not None
            predicted_langs = logprobs.argmax(axis=1)

            for row_idx in range(predicted_langs.shape[0]):
                full_probs.append(
                    logprobs[row_idx, :].tolist()
                )
                full_preds.append(self.lang_spec.lang_names[
                    predicted_langs[row_idx].tolist()
                ])
                full_trues.append(dset_lang_names[
                    batch['lang'][row_idx].tolist()
                ])
                sent_index += 1
            if log is not None:
                log.progress_update(sent_index)
        if log is not None:
            log.progress_end()

        return (full_probs, full_preds, full_trues)

    #########################################
    def _set_gradients(
        self,
        batch: Sequence[Mapping[str, np.ndarray]],
        batch_size: int,
    ) -> None:
        '''
        Set the gradients of the parameters with a single batch of data.

        :param batch: The batch of data which is a pair consisting of the label batch and
            the language batch.
        :param batch_size: The maximum amount of training items to process in one go.
        '''
        if self.get_curr_opt_index() == 0: # Main
            (label_batch, lang_batch) = batch
            num_labels = get_batch_size(label_batch)
            num_langs = get_batch_size(lang_batch)

            for (label_subbatch, lang_subbatch) in itertools.zip_longest(
                get_subbatches(label_batch, batch_size),
                get_subbatches(lang_batch, batch_size),
                fillvalue=None,
            ):
                (label_tensors, lang_tensors) = self.__batch_prepare_inputs(
                    label_subbatch, lang_subbatch
                )
                (label_logits, lang_logits) = self.__batch_logits(label_tensors, lang_tensors)
                assert label_tensors is not None
                assert lang_tensors is not None
                assert label_logits is not None
                assert lang_logits is not None

                label_error = F.cross_entropy(
                    label_logits,
                    label_tensors['label'],
                    reduction='sum',
                )/num_labels

                lang_error = F.cross_entropy(
                    lang_logits,
                    lang_tensors['label'],
                    reduction='sum',
                )/num_langs

                error = label_error + lang_error
                error.backward()

        elif self.get_curr_opt_index() == 1: # Language
            (lang_batch,) = batch
            num_langs = get_batch_size(lang_batch)

            for lang_subbatch in get_subbatches(lang_batch, batch_size):
                (_, lang_tensors) = self.__batch_prepare_inputs(
                    None, lang_subbatch
                )
                (_, lang_logits) = self.__batch_logits(None, lang_tensors)
                assert lang_tensors is not None
                assert lang_logits is not None

                error = F.cross_entropy(
                    lang_logits,
                    lang_tensors['label'],
                    reduction='sum',
                )/num_langs

                error.backward()

        else:
            raise AssertionError()

    #########################################
    def fit(
        self,
        checkpoint_id: str,
        dset_label_train: Dataset[XNLIDataRow],
        dset_lang_train: Dataset[WikipediaDataRow],
        dset_label_val: Dataset[XNLIDataRow],
        dset_lang_val: Dataset[WikipediaDataRow],
        batch_size: int,
        model_path: str,
        hyperparameter_search_mode: bool,
        train_history_path: str,
        checkpoint_manager: CheckpointManager,
        log: Log,
    ) -> None:
        '''
        Train the model using the provided datasets.

        :param checkpoint_id: A unique checkpoint key to use for checkpointing training progress.
        :param dset_label_train: The training set to use to optimise the label
            classification part of the model.
        :param dset_lang_train: The training set to use to optimise the language
            classification part of the model.
        :param dset_label_val: The validation set to use to measure label validation
            performance.
        :param dset_lang_val: The validation set measure language validation performance.
        :param batch_size: The maximum amount of sentences to process in one go.
        :param model_path: The path to the folder that will contain the model pickle files.
            One file called model.pkl will be used to store the best trained model
            (according to the validation set) whilst the other file called checkpoint.pkl
            will be used save a checkpoint model every validation check for resuming
            training in case of interruption.
        :param hyperparameter_search_mode: Whether to enter into hyperparameter search mode where
            minimal output and evaluation is produced.
        :param train_history_path: The path to the folder that will contain the training history.
        :param checkpoint_manager: The checkpoint manager.
        :param log: The log to pass progress information to.
        '''
        if dset_label_train.spec != self.label_spec:
            raise ValueError('Label training set is incompatible with model.')
        if dset_lang_train.spec != self.lang_spec:
            raise ValueError('Language training set is incompatible with model.')
        if dset_label_val.spec != self.label_spec:
            raise ValueError('Label validation set is incompatible with model.')
        if dset_lang_val.spec != self.lang_spec:
            raise ValueError('Language validation set is incompatible with model.')

        label_num_minibatches = dset_label_train.get_num_batches(self.hyperparams.minibatch_size)
        lang_num_minibatches = dset_lang_train.get_num_batches(self.hyperparams.minibatch_size)

        log.log_message('Main phase')
        self.set_curr_opt_index(0)
        ModelTrainingProcess(
            rng=self.rng,
            batch_size=batch_size,
            label_encoder=self.get_label_encodings,
            lang_encoder=self.get_lang_encodings,
            label_predictor=self.get_label_predictions,
            lang_predictor=self.get_lang_predictions,
            dset_label_train=dset_label_train,
            dset_lang_train=dset_lang_train,
            dset_label_val=dset_label_val,
            dset_lang_val=dset_lang_val,
            label_spec=self.label_spec,
            lang_spec=self.lang_spec,
            training_main_module=True,
            hyperparameter_search_mode=hyperparameter_search_mode,
            train_history_path=train_history_path,
            log=log,
        ).run(
            checkpoint_id='{}/main'.format(checkpoint_id),
            model=self,
            batch_size=batch_size,
            minibatch_size=self.hyperparams.minibatch_size,
            num_minibatches=[label_num_minibatches],
            patience=self.hyperparams.patience,
            max_epochs=self.hyperparams.max_epochs,
            model_path=model_path,
            checkpoint_manager=checkpoint_manager,
            log=log,
        )

        if not hyperparameter_search_mode:
            log.log_message('Language phase')
            self.set_curr_opt_index(1)
            with checkpoint_manager.checkpoint('{}/reinit_lang'.format(checkpoint_id)) as handle:
                if handle.was_found_ready():
                    handle.skip()
                obj = self.net.lang_logits.state_dict()
                torch.save(obj, os.path.join(model_path, 'main_phase_lang_module.pkl'))
                self.net.initialise_lang_module(self.hyperparams, self.rng)
                self.net_changed()
            ModelTrainingProcess(
                rng=self.rng,
                batch_size=batch_size,
                label_encoder=self.get_label_encodings,
                lang_encoder=self.get_lang_encodings,
                label_predictor=self.get_label_predictions,
                lang_predictor=self.get_lang_predictions,
                dset_label_train=dset_label_train,
                dset_lang_train=dset_lang_train,
                dset_label_val=dset_label_val,
                dset_lang_val=dset_lang_val,
                label_spec=self.label_spec,
                lang_spec=self.lang_spec,
                training_main_module=False,
                hyperparameter_search_mode=hyperparameter_search_mode,
                train_history_path=train_history_path,
                log=log,
            ).run(
                checkpoint_id='{}/lang'.format(checkpoint_id),
                model=self,
                batch_size=batch_size,
                minibatch_size=self.hyperparams.minibatch_size,
                num_minibatches=[lang_num_minibatches],
                patience=self.hyperparams.patience,
                max_epochs=self.hyperparams.max_epochs,
                model_path=model_path,
                checkpoint_manager=checkpoint_manager,
                log=log,
            )
