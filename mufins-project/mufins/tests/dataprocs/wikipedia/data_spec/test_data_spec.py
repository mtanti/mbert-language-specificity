'''
Unit test for Wikipedia Data Spec class in Wikipedia module.
'''

import unittest
import numpy as np
from mufins.common.tokeniser.tokeniser_mock import TokeniserMock
from mufins.dataprocs.wikipedia.data_spec import WikipediaDataSpec
from mufins.dataprocs.wikipedia.data_row import WikipediaDataRow


#########################################
class TestWikipediaDataSpec(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_no_trim(
        self,
    ) -> None:
        '''
        Test the WikipediaDataSpec class when the text is not trimmed.
        '''
        spec = WikipediaDataSpec(
            10,
            'mock',
            ['en', 'it'],
        )

        self.assertEqual(
            spec.get_field_specs(),
            {
                'tokens': ([10], np.uint32),
                'num_tokens': ([], np.uint8),
                'lang': ([], np.int8),
                'label_mask': ([8], np.bool_),
            }
        )

        row = WikipediaDataRow(
            'John is studying.',
            'en',
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
            'label_mask': np.array(
                [1, 0, 1, 0, 1, 0, 1, 0],
                np.bool_,
            ),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )

        row = WikipediaDataRow(
            'John is studying.',
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
            'num_tokens': np.array(9, np.uint16),
            'lang': np.array(-1, np.int8),
            'label_mask': np.array(
                [1, 0, 1, 0, 1, 0, 1, 0],
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
        Test the WikipediaDataSpec class when the text is trimmed.
        '''
        spec = WikipediaDataSpec(
            5,
            'mock',
            ['en', 'it'],
        )

        self.assertEqual(
            spec.get_field_specs(),
            {
                'tokens': ([5], np.uint32),
                'num_tokens': ([], np.uint8),
                'lang': ([], np.int8),
                'label_mask': ([3], np.bool_),
            }
        )

        row = WikipediaDataRow(
            'John is studying.',
            'en',
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
            'label_mask': np.array(
                [1, 0, 1],
                np.bool_,
            ),
        }

        np.testing.assert_equal(
            spec.preprocess(row),
            preprocessed,
        )

        row = WikipediaDataRow(
            'John is studying.',
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
            'num_tokens': np.array(5, np.uint16),
            'lang': np.array(-1, np.int8),
            'label_mask': np.array(
                [1, 0, 1],
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
