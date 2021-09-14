'''
Unit test for XNLI Data Spec class in xnli module.
'''

import unittest
import numpy as np
from mufins.common.tokeniser.tokeniser_mock import TokeniserMock
from mufins.dataprocs.xnli.data_spec import XNLIDataSpec
from mufins.dataprocs.xnli.data_row import XNLIDataRow


#########################################
class TestXNLIDataSpec(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_no_trim(
        self,
    ) -> None:
        '''
        Test the XNLIDataSpec class when the text is not trimmed.
        '''
        spec = XNLIDataSpec(
            ['en', 'it'],
            18,
            'mock',
            ['contradiction', 'entailment', 'neutral'],
        )

        self.assertEqual(
            spec.get_field_specs(),
            {
                'lang': ([], np.int8),
                'tokens': ([18], np.uint32),
                'num_tokens': ([], np.uint8),
                'label': ([], np.int8),
            }
        )

        row = XNLIDataRow(
            'en',
            'John is studying.',
            'John is swimming.',
            'contradiction',
        )

        preprocessed = {
            'lang': np.array(0, np.int8),
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in (
                        '<CLS> Joh ##n i ##s studyin ##g . <SEP> Joh ##n i ##s swimmin ##g . <SEP>'
                        ' <PAD>'
                    ).split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(17, np.uint16),
            'label': np.array(0, np.int8),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )

        row = XNLIDataRow(
            None,
            'John is studying.',
            'John is swimming.',
            None,
        )

        preprocessed = {
            'lang': np.array(-1, np.int8),
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in (
                        '<CLS> Joh ##n i ##s studyin ##g . <SEP> Joh ##n i ##s swimmin ##g . <SEP>'
                        ' <PAD>'
                    ).split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(17, np.uint16),
            'label': np.array(-1, np.int8),
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
        Test the XNLIDataSpec class when the text is trimmed.
        '''
        spec = XNLIDataSpec(
            ['en', 'it'],
            14,
            'mock',
            ['contradiction', 'entailment', 'neutral'],
        )

        self.assertEqual(
            spec.get_field_specs(),
            {
                'lang': ([], np.int8),
                'tokens': ([14], np.uint32),
                'num_tokens': ([], np.uint8),
                'label': ([], np.int8),
            }
        )

        row = XNLIDataRow(
            'en',
            'John is studying.',
            'John is swimming.',
            'contradiction',
        )

        preprocessed = {
            'lang': np.array(0, np.int8),
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in (
                        '<CLS> Joh ##n i ##s studyin <SEP> Joh ##n i ##s swimmin ##g <SEP>'
                    ).split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(14, np.uint16),
            'label': np.array(0, np.int8),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )

        row = XNLIDataRow(
            None,
            'John is studying.',
            'John is swimming.',
            None,
        )

        preprocessed = {
            'lang': np.array(-1, np.int8),
            'tokens': np.array(
                [
                    TokeniserMock.TOKEN2INDEX[t]
                    for t in (
                        '<CLS> Joh ##n i ##s studyin <SEP> Joh ##n i ##s swimmin ##g <SEP>'
                    ).split(' ')
                ],
                np.uint32,
            ),
            'num_tokens': np.array(14, np.uint16),
            'label': np.array(-1, np.int8),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )


#########################################
if __name__ == '__main__':
    unittest.main()
