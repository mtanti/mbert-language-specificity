#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of MUFINS Project.
'''
Produce a parameter space file consisting of random parameters for hyperparameter tuning.
'''

import os
import argparse
import json
from typing import Mapping, Union, Sequence, Callable, List, cast, Set, Tuple
import jsonschema
import jsonschema.exceptions
import mufins
from mufins.common.random.random_number_generator import RandomNumberGenerator


#########################################
SpecType = Mapping[
    str,
    Mapping[
        str,
        Union[
            bool,
            str,
            int,
            float,
            Sequence[str],
            Sequence[int],
            Sequence[float],
        ]
    ]
]


#########################################
def make_bool_generator(
    rng: RandomNumberGenerator,
) -> Tuple[Callable[[], Union[bool, str, int, float]], int]:
    '''
    Make a random boolean generator.

    :param rng: The random number generator.
    :return: A pair consisting of the generator function and the number of possible values that can
        be generated (or zero if infinite).
    '''
    return (
        lambda: rng.choice([False, True]),
        2,
    )


#########################################
def make_str_generator(
    rng: RandomNumberGenerator,
    values: Sequence[str],
) -> Tuple[Callable[[], Union[bool, str, int, float]], int]:
    '''
    Make a random string generator.

    :param rng: The random number generator.
    :param values: The string values to choose from.
    :return: A pair consisting of the generator function and the number of possible values that can
        be generated (or zero if infinite).
    '''
    return (
        lambda: rng.choice(values),
        len(values),
    )


#########################################
def make_int_generator(
    rng: RandomNumberGenerator,
    values: Sequence[int],
) -> Tuple[Callable[[], Union[bool, str, int, float]], int]:
    '''
    Make a random integer generator.

    :param rng: The random number generator.
    :param values: The integer values to choose from.
    :return: A pair consisting of the generator function and the number of possible values that can
        be generated (or zero if infinite).
    '''
    return (
        lambda: rng.choice(values),
        len(values),
    )


#########################################
def make_float_generator(
    rng: RandomNumberGenerator,
    values: Sequence[float],
) -> Tuple[Callable[[], Union[bool, str, int, float]], int]:
    '''
    Make a random float generator.

    :param rng: The random number generator.
    :param values: The float values to choose from.
    :return: A pair consisting of the generator function and the number of possible values that can
        be generated (or zero if infinite).
    '''
    return (
        lambda: rng.choice(values),
        len(values),
    )


#########################################
def make_int_range_generator(
    rng: RandomNumberGenerator,
    min_value: int,
    max_value: int,
    dist: str,
) -> Tuple[Callable[[], Union[bool, str, int, float]], int]:
    '''
    Make a random ranged integer generator.

    :param rng: The random number generator.
    :param min_value: The minimum value to generate (inclusive).
    :param max_value: The maximum value to generate (inclusive).
    :param dist: he sampling distribution which can be 'uniform' or 'log2'.
    :return: A pair consisting of the generator function and the number of possible values that can
        be generated (or zero if infinite).
    '''
    return (
        lambda: rng.int_range(min_value, max_value, dist),
        max_value - min_value + 1,
    )


#########################################
def make_float_range_generator(
    rng: RandomNumberGenerator,
    min_value: float,
    max_value: float,
    dist: str,
) -> Tuple[Callable[[], Union[bool, str, int, float]], int]:
    '''
    Make a random ranged float generator.

    :param rng: The random number generator.
    :param min_value: The minimum value to generate (inclusive).
    :param max_value: The maximum value to generate (exclusive).
    :param dist: he sampling distribution which can be 'uniform' or 'log10'.
    :return: A pair consisting of the generator function and the number of possible values that can
        be generated (or zero if infinite).
    '''
    return (
        lambda: rng.float_range(min_value, max_value, dist),
        0, # Infinite.
    )


#########################################
def main(
    spec_file_path: str,
    output_file_path: str,
    amount: int,
    seed: int,
) -> None:
    '''
    Produce a parameter space file.

    :param spec_file_path: The path to the JSON file that specifies the valid values of the
        parameters.
    :param output_file_path: The path to the output parameter space file.
    :param amount: The amount of random parameter objects to generate.
    :param seed: The seed to use to randomly generate parameter objects.
    '''
    rng = RandomNumberGenerator(seed)

    with open(
        os.path.join(mufins.path, 'resources', 'json_schemas', 'parameter_space_spec.json'),
        encoding='utf-8',
    ) as f:
        schema = json.load(f)
    jsonschema.Draft7Validator.check_schema(schema)
    validator = jsonschema.Draft7Validator(schema)

    with open(spec_file_path, encoding='utf-8') as f:
        spec: SpecType = json.load(f)
    try:
        validator.validate(spec)
    except jsonschema.exceptions.ValidationError as ex:
        raise ValueError(ex.message) from ex

    search_space_size = 1
    keys = []
    generators: List[Callable[[], Union[str, int, float]]] = list()
    for (key, value) in sorted(spec.items()):
        keys.append(key)
        dtype = cast(str, value['dtype'])
        if dtype == 'bool':
            (gen, size) = make_bool_generator(
                rng,
            )
        elif dtype == 'str':
            (gen, size) = make_str_generator(
                rng,
                cast(Sequence[str], value['values']),
            )
        elif dtype == 'int':
            (gen, size) = make_int_generator(
                rng,
                cast(Sequence[int], value['values']),
            )
        elif dtype == 'float':
            (gen, size) = make_float_generator(
                rng,
                cast(Sequence[float], value['values']),
            )
        elif dtype == 'int_range':
            (gen, size) = make_int_range_generator(
                rng,
                cast(int, value['min']),
                cast(int, value['max']),
                cast(str, value['dist']),
            )
        elif dtype == 'float_range':
            (gen, size) = make_float_range_generator(
                rng,
                cast(float, value['min']),
                cast(float, value['max']),
                cast(str, value['dist']),
            )
        else:
            raise ValueError()

        generators.append(gen)
        search_space_size *= size

    if search_space_size != 0 and search_space_size < amount:
        raise ValueError(
            'The amount of parameters to generate is greater than the search space size.'
        )

    generated: Set[Tuple[Union[str, int, float], ...]] = set()
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for _ in range(amount):
            while True:
                params = tuple(gen() for gen in generators)
                if params not in generated:
                    break
            generated.add(params)

            exp_id = ','.join(
                '{}={}'.format(key, repr(param))
                for (key, param) in zip(keys, params)
            )
            attrs = dict(zip(keys, params))

            print(exp_id, json.dumps(attrs), sep='\t', file=f)


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Produce a parameter space file consisting of random parameters for hyperparameter'
            ' tuning.'
        )
    )
    parser.add_argument(
        '--spec_file_path',
        required=True,
        help='The path to the file that specifies the valid values of the parameters.'
    )
    parser.add_argument(
        '--output_file_path',
        required=True,
        help='The path to the output parameter space file.'
    )
    parser.add_argument(
        '--amount',
        required=True,
        type=int,
        help='The amount of random parameter objects to generate.'
    )
    parser.add_argument(
        '--seed',
        required=True,
        type=int,
        help='The seed to use to randomly generate parameter objects.'
    )

    args = parser.parse_args()
    main(
        spec_file_path=args.spec_file_path,
        output_file_path=args.output_file_path,
        amount=args.amount,
        seed=args.seed,
    )
