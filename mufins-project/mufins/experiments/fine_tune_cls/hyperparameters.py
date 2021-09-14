'''
The hyperparameters of the model.
'''

import json
from typing import Optional, Union, List, Tuple
from mufins.common.hyperparameter.parameter_space import AttributesType, ParameterSpace
from mufins.common.encoder.encoder_factory import ENCODER_NAMES


#########################################
class Hyperparameters():
    '''
    Class with the hyperparameters.
    '''

    #########################################
    def __init__(
        self,
        encoder_name: str,
        layer_index: Optional[int],
        init_stddev: float,
        minibatch_size: int,
        dropout_rate: float,
        freeze_embeddings: bool,
        encoder_learning_rate: Optional[float],
        postencoder_learning_rate: float,
        patience: Optional[int],
        max_epochs: Optional[int],
    ) -> None:
        '''
        Constructor.

        :param encoder_name: The name of the encoder.
        :param layer_index: The layer index of the encoder to use, if supported.
        :param init_stddev: The standard deviation of the random normal number
            generator for the weights of the logits layer.
        :param minibatch_size: The minibatch size to use whilst training.
        :param dropout_rate: The dropout rate used on the dropout layer on top
            of the encoder.
        :param freeze_embeddings: Whether to freeze the encoder's word embeddings or not.
        :param encoder_learning_rate: The gradient descent learning rate of the
            the encoder.
            If None then encoder parameters will be frozen.
        :param postencoder_learning_rate: The gradient descent learning rate of the
            classification layer on top of the encoder.
        :param patience: The number of less than best validation checks to allow
            before terminating training (if not None).
        :param max_epochs: Terminate training on the validation check that happens on or after this
            epoch number (if not None).
        '''
        if encoder_name not in ENCODER_NAMES:
            raise ValueError('Invalid encoder name.')
        self.encoder_name: str = encoder_name

        if layer_index is not None and layer_index < 0:
            raise ValueError('Invalid layer index.')
        self.layer_index: Optional[int] = layer_index

        if init_stddev <= 0.0:
            raise ValueError('Invalid initialisation standard deviation.')
        self.init_stddev: float = init_stddev

        if minibatch_size <= 0:
            raise ValueError('Invalid minibatch size.')
        self.minibatch_size: int = minibatch_size

        if dropout_rate < 0.0:
            raise ValueError('Invalid dropout rate.')
        self.dropout_rate: float = dropout_rate

        self.freeze_embeddings: bool = freeze_embeddings

        if encoder_learning_rate is not None and encoder_learning_rate <= 0.0:
            raise ValueError('Invalid encoder learning rate.')
        self.encoder_learning_rate: Optional[float] = encoder_learning_rate

        if postencoder_learning_rate <= 0.0:
            raise ValueError('Invalid post-encoder learning rate.')
        self.postencoder_learning_rate: float = postencoder_learning_rate

        if patience is not None and patience <= 0:
            raise ValueError('Invalid patience.')
        self.patience: Optional[int] = patience

        if max_epochs is not None and max_epochs < 0:
            raise ValueError('Invalid maximum epochs.')
        self.max_epochs: Optional[int] = max_epochs

    #########################################
    def to_json(
        self,
    ) -> str:
        '''
        Serialise this object to JSON form.

        :return: The JSON string.
        '''
        return json.dumps(dict(
            encoder_name=self.encoder_name,
            layer_index=self.layer_index,
            init_stddev=self.init_stddev,
            minibatch_size=self.minibatch_size,
            dropout_rate=self.dropout_rate,
            freeze_embeddings=self.freeze_embeddings,
            encoder_learning_rate=self.encoder_learning_rate,
            postencoder_learning_rate=self.postencoder_learning_rate,
            patience=self.patience,
            max_epochs=self.max_epochs,
        ), indent=1)

    #########################################
    @staticmethod
    def from_json(
        s: str,
    ) -> 'Hyperparameters':
        '''
        Create an object from a JSON string.

        :param s: The JSON string.
        :return: The object.
        '''
        params = json.loads(s)
        if not isinstance(params, dict):
            raise ValueError('Invalid JSON type.')
        expected_keys = {
            'encoder_name',
            'layer_index',
            'init_stddev',
            'minibatch_size',
            'dropout_rate',
            'freeze_embeddings',
            'encoder_learning_rate',
            'postencoder_learning_rate',
            'patience',
            'max_epochs',
        }
        found_keys = set(params.keys())
        if found_keys != expected_keys:
            raise ValueError('Missing keys: {}, unexpected keys: {}.'.format(
                expected_keys - found_keys,
                found_keys - expected_keys,
            ))

        if not isinstance(params['encoder_name'], (str,)):
            raise ValueError('Invalid encoder name type.')
        if not isinstance(params['layer_index'], (int, type(None))):
            raise ValueError('Invalid layer index type.')
        if not isinstance(params['init_stddev'], (float,)):
            raise ValueError('Invalid initialisation standard deviation type.')
        if not isinstance(params['minibatch_size'], (int,)):
            raise ValueError('Invalid minibatch size type.')
        if not isinstance(params['dropout_rate'], (float,)):
            raise ValueError('Invalid dropout rate type.')
        if not isinstance(params['freeze_embeddings'], (bool,)):
            raise ValueError('Invalid freeze embeddings type.')
        if not isinstance(params['encoder_learning_rate'], (float, type(None))):
            raise ValueError('Invalid encoder learning rate type.')
        if not isinstance(params['postencoder_learning_rate'], (float,)):
            raise ValueError('Invalid post-encoder learning rate type.')
        if not isinstance(params['patience'], (int, type(None))):
            raise ValueError('Invalid patience type.')
        if not isinstance(params['max_epochs'], (int, type(None))):
            raise ValueError('Invalid maximum epochs type.')

        return Hyperparameters(
            encoder_name=params['encoder_name'],
            layer_index=params['layer_index'],
            init_stddev=params['init_stddev'],
            minibatch_size=params['minibatch_size'],
            dropout_rate=params['dropout_rate'],
            freeze_embeddings=params['freeze_embeddings'],
            encoder_learning_rate=params['encoder_learning_rate'],
            postencoder_learning_rate=params['postencoder_learning_rate'],
            patience=params['patience'],
            max_epochs=params['max_epochs'],
        )


#########################################
class FineTuneClsParameterSpace(ParameterSpace[Hyperparameters]):
    '''
    lang_grad_rev_cls parameter space class.
    '''
    # pylint: disable=no-self-use
    # pylint: disable=too-few-public-methods

    #########################################
    def __init__(
        self,
        encoder_name: str,
        layer_index: Optional[int],
        init_stddev: float,
        minibatch_size: int,
        dropout_rate: float,
        freeze_embeddings: bool,
        encoder_learning_rate: Optional[float],
        postencoder_learning_rate: float,
        patience: Optional[int],
        max_epochs: Optional[int],
        attributes_list_or_path: Union[List[Tuple[str, AttributesType]], str],
    ) -> None:
        '''
        Constructor.

        :param encoder_name: The name of the encoder to use.
        :param layer_index: The layer index of the encoder to use, if supported.
        :param init_stddev: The random normal stddev for weights intialisation.
        :param minibatch_size: The minibatch size to use when training.
        :param dropout_rate: The dropout rate to apply to the dropout layer on top
            of the encoder.
        :param freeze_embeddings: Whether to freeze the model's word embeddings.
        :param encoder_learning_rate: The learning rate to use on the encoder.
            If None then encoder parameters will be frozen.
        :param postencoder_learning_rate: The learning rate to use after the encoder.
        :param patience: The number of non-best validation checks to have before
            terminating training (if not None).
        :param max_epochs: Terminate training on the validation check that happens on or after this
            epoch number (if not None).
        :param attributes_list_or_path: Either a list of name-attributes pairs to be converted into
            parameters or a path to a file with said name-attributes pairs.
        '''
        super().__init__(
            dict(
                encoder_name=encoder_name,
                layer_index=layer_index,
                init_stddev=init_stddev,
                minibatch_size=minibatch_size,
                dropout_rate=dropout_rate,
                freeze_embeddings=freeze_embeddings,
                encoder_learning_rate=encoder_learning_rate,
                postencoder_learning_rate=postencoder_learning_rate,
                patience=patience,
                max_epochs=max_epochs,
            ),
            attributes_list_or_path,
        )

    #########################################
    def parameter_decoder_full(
        self,
        full_attributes: AttributesType,
    ) -> Hyperparameters:
        '''
        A function that turns a dictionary of attributes into a complete parameter set object.
        This function is expected to raise an exception in case of an invalid dictionary.

        :param full_attributes: A dictionary of attribute values (with missing values replaced with
            defaults).
        :return: A parameter set.
        '''
        return Hyperparameters.from_json(json.dumps(full_attributes))
