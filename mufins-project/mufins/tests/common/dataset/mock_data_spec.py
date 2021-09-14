'''
Mock data specification for unit tests.
'''

from typing import Mapping, Sequence, Tuple
import numpy as np
from mufins.common.dataset.data_spec import DataSpec


#########################################
class MockDataSpec(DataSpec[str]):
    '''
    Mock data specification class.

    What this specifies is a raw data which is a 4 character string consisting
    of digits. The first 3 digits are stored separately in a 3 element vector
    which is stored in the first field and the last digit is stored in a single
    element vector which is stored in the second field.

    Example:
        '1234' => [[1,2,3], [4]]
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''Empty constructor.'''

    #########################################
    def get_field_specs(
        self,
    ) -> Mapping[str, Tuple[Sequence[int], np.dtype]]:
        '''
        Get a sequence of field specifications that are expected in the
        dataset.

        :return: The field specifications.
        '''
        # pylint: disable=no-self-use
        return {
            'a': ([3], np.int32),
            'b': ([1], np.int32),
        }

    #########################################
    def preprocess(
        self,
        raw: str,
    ) -> Mapping[str, np.ndarray]:
        '''
        Convert raw data into preprocessed numeric arrays, one for each dataset
        field.

        :param raw: A composite object of raw data (such as texts or images).
        :return: A sequence of preprocessed data.
            Note that the batch dimension should not be included in the arrays.
        '''
        # pylint: disable=no-self-use
        return {
            'a': np.array([int(x) for x in raw[0:3]], np.int32),
            'b': np.array([int(x) for x in raw[3:4]], np.int32),
        }
