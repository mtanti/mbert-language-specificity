'''
Preprocessor for XNLI dataset.

This dataset consists of json line files with a label, language, premise, and hypothesis in
each line. The training set is only English and does not have a language field.
'''

import json
from typing import Mapping, Sequence, Tuple, Union
import numpy as np
from mufins.common.dataset.data_spec import DataSpec
from mufins.common.tokeniser.tokeniser import Tokeniser
from mufins.common.tokeniser.tokeniser_factory import tokeniser_factory
from mufins.dataprocs.xnli.data_row import XNLIDataRow


#########################################
class XNLIDataSpec(
    DataSpec[XNLIDataRow]
):
    '''
    Data specification for the XTREME XNLI dataset.
    '''

    #########################################
    def __init__(
        self,
        lang_names: Sequence[str],
        max_num_tokens: int,
        tokeniser_name: str,
        label_names: Sequence[str],
    ) -> None:
        '''
        Constructor.

        :param lang_names: The names of all languages used.
        :param max_num_tokens: The maximum sentence length in terms of tokens beyond which is
            trimmed.
        :param tokeniser_name: The tokeniser to use.
        :param label_names: The names of the different labels.
        '''
        self.max_num_tokens: int = max_num_tokens
        self.tokeniser_name: str = tokeniser_name
        self.tokeniser: Tokeniser = tokeniser_factory(tokeniser_name)
        self.lang_names: Sequence[str] = sorted(set(lang_names))
        self.lang_to_index: Mapping[str, int] = {
            lang: i for (i, lang) in enumerate(lang_names)
        }
        self.label_names: Sequence[str] = sorted(set(label_names))
        self.label_to_index: Mapping[str, int] = {
            label: i for (i, label) in enumerate(self.label_names)
        }

    #########################################
    def __eq__(
        self,
        other: object,
    ) -> bool:
        '''
        Check if this data spec is equal to another data spec.

        :param other: The other object.
        :return: Whether they are equal.
        '''
        if isinstance(other, self.__class__):
            return all([
                self.max_num_tokens == other.max_num_tokens,
                self.tokeniser_name == other.tokeniser_name,
                self.lang_names == other.lang_names,
                self.label_names == other.label_names,
            ])
        return False

    #########################################
    def to_json(
        self,
    ) -> str:
        '''
        Serialise this object to JSON form.

        :return: The JSON string.
        '''
        return json.dumps(dict(
            lang_names=self.lang_names,
            max_num_tokens=self.max_num_tokens,
            tokeniser_name=self.tokeniser_name,
            label_names=self.label_names,
        ), indent=1)

    #########################################
    @staticmethod
    def from_json(
        s: str,
    ) -> 'XNLIDataSpec':
        '''
        Create an object from a JSON string.

        :param s: The JSON string.
        :return: The object.
        '''
        params = json.loads(s)
        if not isinstance(params, dict):
            raise ValueError('Invalid JSON type.')
        expected_keys = {
            'lang_names',
            'max_num_tokens',
            'tokeniser_name',
            'label_names',
        }
        found_keys = set(params.keys())
        if found_keys != expected_keys:
            raise ValueError('Missing keys: {}, unexpected keys: {}.'.format(
                expected_keys - found_keys,
                found_keys - expected_keys,
            ))

        if (
            not isinstance(params['lang_names'], (list,))
            or not all(isinstance(x, str) for x in params['lang_names'])
        ):
            raise ValueError('Invalid language names type.')
        if not isinstance(params['max_num_tokens'], (int,)):
            raise ValueError('Invalid maximum number of tokens type.')
        if not isinstance(params['tokeniser_name'], (str,)):
            raise ValueError('Invalid tokeniser name type.')
        if (
            not isinstance(params['label_names'], (list,))
            or not all(isinstance(x, str) for x in params['label_names'])
        ):
            raise ValueError('Invalid label names type.')

        return XNLIDataSpec(
            lang_names=params['lang_names'],
            max_num_tokens=params['max_num_tokens'],
            tokeniser_name=params['tokeniser_name'],
            label_names=params['label_names'],
        )

    #########################################
    def get_field_specs(
        self,
    ) -> Mapping[str, Tuple[Sequence[int], Union[np.dtype, str]]]:
        '''
        Get a mapping of name - array specifications that are expected in the
        dataset.

        :return: The field specifications.
        '''
        return {
            'lang': ([], np.int8),
            'tokens': ([self.max_num_tokens], np.uint32),
            'num_tokens': ([], np.uint8),
            'label': ([], np.int8),
        }

    #########################################
    def preprocess(
        self,
        raw: XNLIDataRow,
    ) -> Mapping[str, np.ndarray]:
        '''
        Convert raw data into preprocessed numeric arrays, one for each dataset
        field.

        :param raw: A composite object of raw data (such as texts or images).
        :return: A mapping of field names to preprocessed data.
            Note that the batch dimension should not be included in the arrays.
        '''
        if raw.lang is None:
            lang_index = -1
        else:
            lang_index = self.lang_to_index[raw.lang]

        token_indexes = self.tokeniser.indexify_text_pair(
            raw.premise_text,
            raw.hypothesis_text,
            self.max_num_tokens,
        )
        padded_token_indexes = np.zeros([self.max_num_tokens], np.uint32)
        num_tokens = len(token_indexes)
        padded_token_indexes[:num_tokens] = token_indexes

        label_index = -1
        if raw.label is not None:
            label_index = self.label_to_index[raw.label]

        return {
            'lang': np.array(lang_index, np.int8),
            'tokens': padded_token_indexes,
            'num_tokens': np.array(num_tokens, np.uint8),
            'label': label_index,
        }
