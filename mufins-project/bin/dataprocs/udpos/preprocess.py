#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of MUFINS Project.
'''
UDPOS preprocessor.
'''

import os
import argparse
from mufins.dataprocs.udpos.preprocess import udpos_preprocess


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess the UDPOS dataset.'
    )
    parser.add_argument(
        '--src_path',
        required=True,
        help='The path to the raw UDPOS dataset.'
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
        help='The fraction of English rows in the original training set to be training data.'
    )
    parser.add_argument(
        '--val_fraction',
        required=True,
        type=float,
        help='The fraction of English rows in the original training set to be validation data.'
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
    udpos_preprocess(
        src_path=os.path.abspath(args.src_path),
        dst_path=os.path.abspath(args.dst_path),
        tokeniser_name=args.tokeniser_name,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        max_num_tokens=args.max_num_tokens,
        seed=args.seed,
        verbose=args.verbose == 'yes',
        debug_mode=args.debug_mode == 'yes',
    )
