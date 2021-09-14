'''
Unit test for UDPOS Data Spec class in udpos module.
'''

import unittest
import numpy as np
from mufins.common.tokeniser.tokeniser_mock import TokeniserMock
from mufins.dataprocs.udpos.data_spec import UDPOSDataSpec
from mufins.dataprocs.udpos.data_row import UDPOSDataRow


#########################################
class TestUDPOSDataSpec(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_no_trim(
        self,
    ) -> None:
        '''
        Test the UDPOSDataSpec class when the text is not trimmed.
        '''
        spec = UDPOSDataSpec(
            10,
            'mock',
            ['en', 'it'],
            ['NOUN', 'PUNC', 'VERB'],
        )

        self.assertEqual(
            spec.get_field_specs(),
            {
                'tokens': ([10], np.uint32),
                'num_tokens': ([], np.uint8),
                'lang': ([], np.int8),
                'labels': ([8], np.int8),
                'label_mask': ([8], np.bool_),
            }
        )

        row = UDPOSDataRow(
            'John is studying .'.split(' '),
            'en',
            'NOUN VERB NOUN PUNC'.split(' '),
        )

        preprocessed = {
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in '<CLS> Joh ##n i ##s studyin ##g . <SEP> <PAD>'.split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(9, np.uint16),
            'lang': np.array(0, np.int8),
            'labels': np.array(
                [0, 0, 2, 2, 0, 0, 1, 0],
                np.int8,
            ),
            'label_mask': np.array(
                [1, 0, 1, 0, 1, 0, 1, 0],
                np.bool_,
            ),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )

        row = UDPOSDataRow(
            'John is studying .'.split(' '),
            None,
            None,
        )

        preprocessed = {
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in '<CLS> Joh ##n i ##s studyin ##g . <SEP> <PAD>'.split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(9, np.uint8),
            'lang': np.array(-1, np.int8),
            'labels': np.array(
                [-1, -1, -1, -1, -1, -1, -1, 0],
                np.int8,
            ),
            'label_mask': np.array(
                [0, 0, 0, 0, 0, 0, 0, 0],
                np.bool_,
            ),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )

    #########################################
    def test_trim(
        self,
    ) -> None:
        '''
        Test the UDPOSDataSpec class when the text is trimmed.
        '''
        spec = UDPOSDataSpec(
            5,
            'mock',
            ['en', 'it'],
            ['NOUN', 'PUNC', 'VERB'],
        )

        self.assertEqual(
            spec.get_field_specs(),
            {
                'tokens': ([5], np.uint32),
                'num_tokens': ([], np.uint8),
                'lang': ([], np.int8),
                'labels': ([3], np.int8),
                'label_mask': ([3], np.bool_),
            }
        )

        row = UDPOSDataRow(
            'John is studying .'.split(' '),
            'en',
            'NOUN VERB NOUN PUNC'.split(' '),
        )

        preprocessed = {
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in '<CLS> Joh ##n i <SEP>'.split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(5, np.uint16),
            'lang': np.array(0, np.int8),
            'labels': np.array(
                [0, 0, 2],
                np.int8,
            ),
            'label_mask': np.array(
                [1, 0, 1],
                np.bool_,
            ),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )

        row = UDPOSDataRow(
            'John is studying .'.split(' '),
            None,
            None,
        )

        preprocessed = {
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in '<CLS> Joh ##n i <SEP>'.split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(5, np.uint8),
            'lang': np.array(-1, np.int8),
            'labels': np.array(
                [-1, -1, -1],
                np.int8,
            ),
            'label_mask': np.array(
                [0, 0, 0],
                np.bool_,
            ),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )


#########################################
if __name__ == '__main__':
    unittest.main()
