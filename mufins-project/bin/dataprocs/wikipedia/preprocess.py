#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of MUFINS Project.
'''
Wikipedia preprocessor.
'''

import os
import argparse
from mufins.dataprocs.wikipedia.preprocess import wikipedia_preprocess


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess the Wikipedia dataset.'
    )
    parser.add_argument(
        '--src_path',
        required=True,
        help='The path to the raw Wikipedia dataset.'
    )
    parser.add_argument(
        '--dst_path',
        required=True,
        help='The path to the outputted processed dataset.'
    )
    parser.add_argument(
        '--tokeniser_name',
        required=True,
        help='The tokeniser to use.'
    )
    parser.add_argument(
        '--train_fraction',
        required=True,
        type=float,
        help='The fraction of rows in the data set to be training data.'
    )
    parser.add_argument(
        '--val_fraction',
        required=True,
        type=float,
        help='The fraction of rows in the data set to be validation data.'
    )
    parser.add_argument(
        '--dev_fraction',
        required=True,
        type=float,
        help='The fraction of rows in the data set to be development data.'
    )
    parser.add_argument(
        '--test_fraction',
        required=True,
        type=float,
        help='The fraction of rows in the data set to be test data.'
    )
    parser.add_argument(
        '--max_num_texts_per_lang',
        required=True,
        type=int,
        help='The maximum number of texts per language to keep before splitting.'
    )
    parser.add_argument(
        '--min_num_chars',
        required=True,
        type=int,
        help='The minimum number of characters in a text allowed to include in the data set.'
    )
    parser.add_argument(
        '--max_num_tokens',
        required=True,
        type=int,
        help='The maximum number of tokens in a sentence beyond which is trimmed.'
    )
    parser.add_argument(
        '--seed',
        required=False,
        type=int,
        default=None,
        help='The seed to use for generating randomness.'
    )
    parser.add_argument(
        '--verbose',
        required=False,
        choices=['yes', 'no'],
        default='yes',
        help='Whether to display progress information.'
    )
    parser.add_argument(
        '--debug_mode',
        required=False,
        choices=['yes', 'no'],
        default='yes',
        help='Whether to display full error information.'
    )

    args = parser.parse_args()
    wikipedia_preprocess(
        src_path=os.path.abspath(args.src_path),
        dst_path=os.path.abspath(args.dst_path),
        tokeniser_name=args.tokeniser_name,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
        max_num_texts_per_lang=args.max_num_texts_per_lang,
        min_num_chars=args.min_num_chars,
        max_num_tokens=args.max_num_tokens,
        seed=args.seed,
        verbose=args.verbose == 'yes',
        debug_mode=args.debug_mode == 'yes',
    )
