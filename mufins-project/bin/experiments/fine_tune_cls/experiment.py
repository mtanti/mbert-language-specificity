#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of MUFINS Project.
'''
fine_tune_cls experiment.
'''

import os
import argparse
from mufins.experiments.fine_tune_cls.experiment import (
    FineTuneClsParameterSpace, fine_tune_cls_experiment,
)


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the experiment coded fine_tune_cls.'
    )
    parser.add_argument(
        '--label_src_path',
        required=True,
        help='The path to the preprocessed label dataset.'
    )
    parser.add_argument(
        '--lang_src_path',
        required=True,
        help='The path to the preprocessed language dataset.'
    )
    parser.add_argument(
        '--dst_path',
        required=True,
        help='The path to the outputted experiment results.'
    )
    parser.add_argument(
        '--device_name',
        required=True,
        help='The GPU device name to use such as \'cuda:0\'. Use \'cpu\' if no' +
            ' GPU is available.'
    )
    parser.add_argument(
        '--default_encoder_name',
        required=True,
        help='The name of the encoder to use.'
    )
    parser.add_argument(
        '--default_layer_index',
        required=False,
        type=int,
        default=None,
        help='The encoder\'s layer index to use, if supported.'
    )
    parser.add_argument(
        '--default_init_stddev',
        required=True,
        type=float,
        help='The random normal stddev for weights intialisation.'
    )
    parser.add_argument(
        '--default_minibatch_size',
        required=True,
        type=int,
        help='The minibatch size to use when training.'
    )
    parser.add_argument(
        '--default_dropout_rate',
        required=True,
        type=float,
        help='The dropout rate to apply to the dropout layer on top of encoder.'
    )
    parser.add_argument(
        '--default_freeze_embeddings',
        required=True,
        choices=['yes', 'no'],
        default='yes',
        help='Whether to freeze the model\'s word embeddings.'
    )
    parser.add_argument(
        '--default_encoder_learning_rate',
        required=False,
        default=None,
        type=float,
        help='The learning rate to use on the encoder.'
    )
    parser.add_argument(
        '--default_postencoder_learning_rate',
        required=True,
        type=float,
        help='The learning rate to use on the layers after the encoder.'
    )
    parser.add_argument(
        '--default_patience',
        required=False,
        default=None,
        type=int,
        help=(
            'The number of non-best validation checks in sequence to have before terminating'
            ' training.'
        )
    )
    parser.add_argument(
        '--default_max_epochs',
        required=False,
        default=None,
        type=int,
        help='End training after this epoch number.'
    )
    parser.add_argument(
        '--parameter_space_path',
        required=True,
        help='The path to the parameter space file.'
    )
    parser.add_argument(
        '--hyperparameter_search_mode',
        required=False,
        choices=['yes', 'no'],
        default='yes',
        help=(
            'Whether to run in hyperparameter search mode where the dev set is used instead'
            ' of the test set and minimal output and evaluation is produced.'
        )
    )
    parser.add_argument(
        '--batch_size',
        required=True,
        type=int,
        help='The maximum number of data rows to process at once.'
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
    fine_tune_cls_experiment(
        label_src_path=os.path.abspath(args.label_src_path),
        lang_src_path=os.path.abspath(args.lang_src_path),
        dst_path=os.path.abspath(args.dst_path),
        device_name=args.device_name,
        parameter_space=FineTuneClsParameterSpace(
            encoder_name=args.default_encoder_name,
            layer_index=args.default_layer_index,
            init_stddev=args.default_init_stddev,
            minibatch_size=args.default_minibatch_size,
            dropout_rate=args.default_dropout_rate,
            freeze_embeddings=args.default_freeze_embeddings == 'yes',
            encoder_learning_rate=args.default_encoder_learning_rate,
            postencoder_learning_rate=args.default_postencoder_learning_rate,
            patience=args.default_patience,
            max_epochs=args.default_max_epochs,

            attributes_list_or_path=os.path.abspath(args.parameter_space_path),
        ),
        hyperparameter_search_mode=args.hyperparameter_search_mode == 'yes',
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=args.verbose == 'yes',
        debug_mode=args.debug_mode == 'yes',
    )
