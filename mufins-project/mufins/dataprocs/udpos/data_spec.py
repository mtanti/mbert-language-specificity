'''
Preprocessor for UDPOS tagging dataset.

This data consists of CONLLU formatted text files which put each word on a separate line with its
corresponding tag.
'''

import json
from typing import Mapping, Sequence, Tuple, Union
import numpy as np
from mufins.common.tokeniser.tokeniser import Tokeniser
from mufins.common.tokeniser.tokeniser_factory import tokeniser_factory
from mufins.common.dataset.data_spec import DataSpec
from mufins.dataprocs.udpos.data_row import UDPOSDataRow


#########################################
class UDPOSDataSpec(
    DataSpec[UDPOSDataRow]
):
    '''
    Data specification for the UDPOS dataset.
    '''

    #########################################
    def __init__(
        self,
        max_num_tokens: int,
        tokeniser_name: str,
        lang_names: Sequence[str],
        label_names: Sequence[str],
    ) -> None:
        '''
        Constructor.

        :param max_num_tokens: The maximum sentence length in terms of tokens beyond which is
            trimmed.
        :param tokeniser_name: The tokeniser to use.
        :param lang_names: The names of all languages used.
        :param label_names: The names of the different token labels.
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
            max_num_tokens=self.max_num_tokens,
            tokeniser_name=self.tokeniser_name,
            lang_names=self.lang_names,
            label_names=self.label_names,
        ), indent=1)

    #########################################
    @staticmethod
    def from_json(
        s: str,
    ) -> 'UDPOSDataSpec':
        '''
        Create an object from a JSON string.

        :param s: The JSON string.
        :return: The object.
        '''
        params = json.loads(s)
        if not isinstance(params, dict):
            raise ValueError('Invalid JSON type.')
        expected_keys = {
            'max_num_tokens',
            'tokeniser_name',
            'lang_names',
            'label_names',
        }
        found_keys = set(params.keys())
        if found_keys != expected_keys:
            raise ValueError('Missing keys: {}, unexpected keys: {}.'.format(
                expected_keys - found_keys,
                found_keys - expected_keys,
            ))

        if not isinstance(params['max_num_tokens'], (int,)):
            raise ValueError('Invalid maximum number of tokens type.')
        if not isinstance(params['tokeniser_name'], (str,)):
            raise ValueError('Invalid tokeniser name type.')
        if (
            not isinstance(params['lang_names'], (list,))
            or not all(isinstance(x, str) for x in params['lang_names'])
        ):
            raise ValueError('Invalid language names type.')
        if (
            not isinstance(params['label_names'], (list,))
            or not all(isinstance(x, str) for x in params['label_names'])
        ):
            raise ValueError('Invalid label names type.')

        return UDPOSDataSpec(
            max_num_tokens=params['max_num_tokens'],
            tokeniser_name=params['tokeniser_name'],
            lang_names=params['lang_names'],
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
            'tokens': ([self.max_num_tokens], np.uint32),
            'num_tokens': ([], np.uint8),
            'lang': ([], np.int8),
            'labels': ([self.max_num_tokens - self.tokeniser.noncontent_total], np.int8),
            'label_mask': ([self.max_num_tokens - self.tokeniser.noncontent_total], np.bool_),
        }

    #########################################
    def preprocess(
        self,
        raw: UDPOSDataRow,
    ) -> Mapping[str, np.ndarray]:
        '''
        Convert raw data into preprocessed numeric arrays, one for each dataset
        field.

        :param raw: A composite object of raw data (such as texts or images).
        :return: A mapping of field names to preprocessed data.
            Note that the batch dimension should not be included in the arrays.
        '''
        slc = self.tokeniser.content_slice
        noncontent = self.tokeniser.noncontent_total

        token_indexes = self.tokeniser.indexify_text(
            ' '.join(raw.words),
            self.max_num_tokens,
        )
        padded_token_indexes = np.zeros([self.max_num_tokens], np.uint32)
        num_tokens = len(token_indexes)
        padded_token_indexes[:num_tokens] = token_indexes

        if raw.lang is None:
            lang_index = -1
        else:
            lang_index = self.lang_to_index[raw.lang]

        padded_label_indexes = np.zeros([self.max_num_tokens - noncontent], np.int8)
        padded_label_mask = np.zeros([self.max_num_tokens - noncontent], np.bool_)
        (tokens, is_fronts) = self.tokeniser.textify_indexes(token_indexes[slc])
        label_index = 0
        curr_word = raw.words[label_index]
        if raw.labels is None:
            curr_label = -1
        else:
            curr_label = self.label_to_index[raw.labels[label_index]]
        for (i, (mask, token)) in enumerate(zip(is_fronts, tokens)):
            if raw.labels is None:
                padded_label_indexes[i] = -1
            else:
                padded_label_mask[i] = mask
                padded_label_indexes[i] = curr_label

                curr_word = curr_word.strip()
                assert curr_word.startswith(token)
                curr_word = curr_word[len(token):]
                if curr_word == '':
                    label_index += 1
                    if label_index == len(raw.words):
                        assert i == len(tokens) - 1
                        break
                    curr_word = raw.words[label_index]
                    curr_label = self.label_to_index[raw.labels[label_index]]

        return {
            'lang': np.array(lang_index, np.int8),
            'tokens': padded_token_indexes,
            'num_tokens': np.array(num_tokens, np.uint8),
            'labels': padded_label_indexes,
            'label_mask': padded_label_mask,
        }
