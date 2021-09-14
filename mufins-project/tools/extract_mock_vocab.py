#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of MUFINS Project.
'''
Extract a vocabulary of known tokens from all of the text in the mock data sets for use in
TokeniserMock.
'''

import os
import argparse
import json
import glob
from typing import Set
import mufins
from mufins.common.tokeniser.tokeniser_mock import TokeniserMock


#########################################
def extract_vocab(
) -> Set[str]:
    '''
    Extract a vocabulary.

    :return: The vocabulary.
    '''
    tokens: Set[str] = set()
    base_path = os.path.join(mufins.path, 'tests', 'dataprocs')

    for path in [
        os.path.join(
            base_path, 'cityf', 'ner', 'mock_dataset', 'nlu_en.json1'
        ),
        os.path.join(
            base_path, 'cityf', 'ner', 'mock_dataset', 'nlu_it.json1'
        ),
    ]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                for token in TokeniserMock.tokenise(json.loads(line)['text']):
                    tokens.add(token)

    for path in [
        os.path.join(
            base_path, 'udpos', 'mock_dataset', 'ud-treebanks-v2.7', 'UD_English-A',
            'en_a-ud-test.conllu'
        ),
        os.path.join(
            base_path, 'udpos', 'mock_dataset', 'ud-treebanks-v2.7', 'UD_English-B',
            'en_b-ud-train.conllu'
        ),
        os.path.join(
            base_path, 'udpos', 'mock_dataset', 'ud-treebanks-v2.7', 'UD_English-B',
            'en_b-ud-dev.conllu'
        ),
        os.path.join(
            base_path, 'udpos', 'mock_dataset', 'ud-treebanks-v2.7', 'UD_Italian-A',
            'it_a-ud-test.conllu'
        ),
        os.path.join(
            base_path, 'udpos', 'mock_dataset', 'ud-treebanks-v2.7', 'UD_Italian-B',
            'it_b-ud-train.conllu'
        ),
        os.path.join(
            base_path, 'udpos', 'mock_dataset', 'ud-treebanks-v2.7', 'UD_Italian-B',
            'it_b-ud-dev.conllu'
        ),
    ]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                for token in TokeniserMock.tokenise(line.split('\t')[1]):
                    tokens.add(token)

    for path in glob.glob(
        os.path.join(
            base_path, 'wikipedia', 'mock_dataset', '*', '*', '*.txt'
        )
    ):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                for token in TokeniserMock.tokenise(line):
                    tokens.add(token)

    for path in [
        os.path.join(
            base_path, 'xnli', 'mock_dataset', 'multinli_1.0', 'multinli_1.0_train.jsonl'
        ),
        os.path.join(
            base_path, 'xnli', 'mock_dataset', 'XNLI-1.0', 'xnli.dev.jsonl'
        ),
        os.path.join(
            base_path, 'xnli', 'mock_dataset', 'XNLI-1.0', 'xnli.test.jsonl'
        ),
    ]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                for token in TokeniserMock.tokenise(json.loads(line)['sentence1']):
                    tokens.add(token)
                for token in TokeniserMock.tokenise(json.loads(line)['sentence2']):
                    tokens.add(token)

    return tokens


#########################################
def print_vocab(
    vocab: Set[str],
) -> None:
    '''
    Print the extracted vocabulary in a suitable format.

    :param vocab: The vocabulary.
    '''
    vocab_ = '<PAD> <SEP> <CLS> <UNK>'.split(' ') + sorted(vocab)
    width = 10
    for i in range(0, len(vocab_), width):
        if i == 0:
            line = '\'{}\' # {}-{}'.format(' '.join(vocab_[i:i+width]), i, i + width - 1)
        else:
            line = '+ \' {}\' # {}-{}'.format(' '.join(vocab_[i:i+width]), i, i + width - 1)
        print(line)


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract a mock vocabulary from the mock data sets and print it out.'
    )

    parser.parse_args()

    print_vocab(extract_vocab())
