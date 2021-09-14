'''
Parameter space class for generating a sequence of parameters to evaluate during
experimentation.

Experiments consist of systematically changing parameters and measuring the resulting performance.
The set of different parameter set objects that are explored are called a parameter space. Each
paramater set object is accompanied by an experiment ID, a unique string that is used for
systematically and briefly describing what the parameter set object represents. Experiment IDs, or
just IDs, in addition to being unique must be also be valid filenames.

In order to make generating parameter set objects easier, the parameter space class allows you to
provide a list of name-attribute dictionaries called attributes objects which are then converted
into parameter set objects. The attributes objects need not correspond to the parameter set objects
directly, and in fact are expected to only include the minimum changing information between
parameters. Any parameters that are constant across all the parameter space can be specified once as
default attributes. Attributes objects have the advantage of being possible to store in plain text
files as JSON encoded objects.

A parameter space file can be used to load the parameter space from. Each line in the text file
consists of the following format:
<experiment ID> <tab> <JSON encoded attributes object>
'''

import json
import re
from abc import ABC, abstractmethod
from typing import Generic, Iterator, List, Mapping, Tuple, TypeVar, Union
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException


#########################################
T = TypeVar('T')
AttributesType = Mapping[str, Union[int, float, str, bool, None]]

class ParameterSpace(ABC, Generic[T]):
    '''
    Parameter space class.

    T is a generic for the parameter set object that is an element of the
    parameter space. It can be generated from a dictionary of attribute values.
    '''

    #########################################
    @staticmethod
    def __load_file(
        path: str,
    ) -> List[Tuple[str, AttributesType]]:
        '''
        Load the parameter space file.

        :param path: The path to the file.
        :return: The attributes list
        '''
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

        attributes_list: List[Tuple[str, AttributesType]] = list()
        for (line_num, line) in enumerate(lines, 1):
            fields = line.split('\t')
            if len(fields) != 2:
                raise IncompatibleExistingDataException(
                    'Invalid line in parameters file at line {}: Line is not '
                    'two fields separated by a tab.'.format(line_num)
                )

            exp_id = fields[0]
            if len(exp_id) == 0:
                raise IncompatibleExistingDataException(
                    'Invalid line in parameters file at line {}: First field '
                    'is empty.'.format(line_num)
                )

            try:
                attributes = json.loads(fields[1])
            except json.decoder.JSONDecodeError as ex:
                raise IncompatibleExistingDataException(
                    'Invalid line in parameters file at line {}: Second field '
                    'is not a valid JSON encoded object. Error reported: '
                    '{}'.format(line_num, str(ex))
                ) from None
            if not isinstance(attributes, dict):
                raise IncompatibleExistingDataException(
                    'Invalid line in parameters file at line {}: Second field '
                    'does not encode a dictionary object. Found: '
                    '{}'.format(line_num, str(type(attributes)))
                )
            for (key, value) in attributes.items():
                if not re.fullmatch(r'[a-z_]+', key):
                    raise IncompatibleExistingDataException(
                        'Invalid line in parameters file at line {}: Attribute {} '
                        'is not a valid key.'.format(line_num, key)
                    )
                if not any([
                    isinstance(value, int),
                    isinstance(value, float),
                    isinstance(value, str),
                    isinstance(value, bool),
                    value is None,
                ]):
                    raise IncompatibleExistingDataException(
                        'Invalid line in parameters file at line {}: Attribute {} '
                        'is not a valid type. Found: '
                        '{}'.format(line_num, key, str(type(value)))
                    )

            attributes_list.append((exp_id, attributes))

        return attributes_list

    #########################################
    def __init__(
        self,
        default_attributes: AttributesType,
        attributes_list_or_path: Union[List[Tuple[str, AttributesType]], str],
    ) -> None:
        '''
        Constructor.

        :param default_attributes: The default values for missing attributes.
        :param attributes_list_or_path: Either a list of id-attributes pairs to be converted into
            parameters or a path to a file with said id-attributes pairs.
        '''
        self.default_attributes: AttributesType = default_attributes
        self.attributes_list: List[Tuple[str, AttributesType]] = list()

        if isinstance(attributes_list_or_path, str):
            self.attributes_list = self.__load_file(attributes_list_or_path)
        else:
            self.attributes_list = attributes_list_or_path

        # Validate attributes list.
        seen_ids = set()
        invalid_ids = {'.', '..'}
        invalid_id_chars = set('\\/?"<>|*:\n\t')
        for (exp_id, attributes) in self.attributes_list:
            if exp_id in invalid_ids:
                raise ValueError('ID {} is not valid.'.format(exp_id))
            if exp_id.startswith(' ') or exp_id.endswith(' '):
                raise ValueError('ID {} cannot start or end with spaces.'.format(exp_id))
            if len(set(exp_id) & invalid_id_chars) > 0:
                raise ValueError(
                    'ID {} contains invalid characters: {:r}.'.format(
                        exp_id,
                        set(exp_id) & invalid_id_chars,
                    )
                )
            if exp_id in seen_ids:
                raise ValueError('Duplicate ID {} found.'.format(exp_id))
            seen_ids.add(exp_id)

            try:
                self.parameter_decoder(attributes)
            except Exception as ex:
                raise ValueError(
                    'ID {} resulted in the following error when decoding:\n{}.'.format(
                        exp_id, ex
                    )
                ) from ex

    #########################################
    def parameter_decoder(
        self,
        partial_attributes: AttributesType,
    ) -> T:
        '''
        A function that turns a dictionary of attributes into a parameter set object.
        The attributes dictionary does not need to define all required attributes as
        they will be copied over from the default attributes.

        :param partial_attributes: A dictionary of attribute values.
        :return: A parameter set.
        '''
        full_attributes = dict(self.default_attributes)
        full_attributes.update(partial_attributes)
        return self.parameter_decoder_full(full_attributes)

    #########################################
    @abstractmethod
    def parameter_decoder_full(
        self,
        full_attributes: AttributesType,
    ) -> T:
        '''
        A function that turns a dictionary of attributes into a parameter set object.
        The attributes dictionary must have all required attributes defined.
        This function is expected to raise an exception in case of an invalid dictionary.

        :param full_attributes: A dictionary of attribute values (with default values being
            used in place of missing values).
        :return: A parameter set.
        '''

    #########################################
    def __iter__(
        self,
    ) -> Iterator[Tuple[str, T]]:
        '''
        Iterate over the parameter space.

        :return: An iterator of tuples consisting of id - parameter
            set object pairs.
        '''
        for (name, attributes) in self.attributes_list:
            yield (name, self.parameter_decoder(attributes))
