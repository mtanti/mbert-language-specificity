'''
Random number generator.
'''

import random
import math
from typing import Optional, List, Sequence, TypeVar
import numpy as np


#########################################
T = TypeVar('T')

class RandomNumberGenerator():
    '''
    Provide all the random features required by the program.
    '''

    #########################################
    def __init__(
        self,
        seed: Optional[int] = None,
    ) -> None:
        '''
        Constructor.

        :param seed: The random seed to used.
            If None then a random seed will be used.
        '''
        if seed is None:
            seed = RandomNumberGenerator.make_seed()
        self.seed: int = seed
        self.rng: random.Random = random.Random(seed)

    #########################################
    @staticmethod
    def make_seed(
    ) -> int:
        '''
        Return a random seed to use.

        :return: A 64-bit integer.
        '''
        return random.getrandbits(64)

    #########################################
    def get_child(
        self,
    ) -> 'RandomNumberGenerator':
        '''
        Get another random number generator with a randomly generated seed.

        :return: The random number generator object.
        '''
        return RandomNumberGenerator(self.rng.getrandbits(64))

    #########################################
    def index(
        self,
        seq: Sequence[T],
    ) -> int:
        '''
        Choose the index of a random item in a sequence.

        :param seq: The sequence to choose from.
        :return: The index.
        '''
        return self.rng.randrange(len(seq))

    #########################################
    def choice(
        self,
        seq: Sequence[T],
    ) -> T:
        '''
        Choose one item out of all the items in a sequence.

        :param seq: The sequence to choose from.
        :return: The item.
        '''
        return self.rng.choice(seq)

    #########################################
    def shuffle(
        self,
        seq: List[T],
    ) -> None:
        '''
        Shuffle the items in a sequence.

        :param seq: The sequence to shuffle.
        '''
        self.rng.shuffle(seq)

    #########################################
    def array_normal(
        self,
        mean: float,
        stddev: float,
        shape: Sequence[int],
    ) -> np.ndarray:
        '''
        Generator a random numpy array using normal random numbers.

        :param mean: The mean of the normal distribution.
        :param stddev: The standard deviation of the normal distribution.
        :param shape: The numpy shape of the array.
        :return: The array.
        '''
        result = np.empty(shape, np.float32)
        with np.nditer(result, op_flags=['readwrite']) as nditer:
            for x in nditer:
                x[...] = self.rng.gauss(mean, stddev)
        return result

    #########################################
    def int_range(
        self,
        min_value: int,
        max_value: int,
        dist: str,
    ) -> int:
        '''
        Choose a random integer from a range using a distribution.

        :param min_value: The range minimum (inclusive).
        :param max_value: The range maximum (inclusive).
        :param dist: The sampling distribution which can be 'uniform' or 'log2'.
        :return: The random number.
        '''
        if dist == 'uniform':
            return self.rng.randrange(min_value, max_value + 1)
        if dist == 'log2':
            return round(2**self.rng.uniform(math.log2(min_value), math.log2(max_value)))
        raise ValueError('Distribution must be \'uniform\' or \'log2\'.')

    #########################################
    def float_range(
        self,
        min_value: float,
        max_value: float,
        dist: str,
    ) -> float:
        '''
        Choose a random float from a range using a distribution.

        :param min_value: The range minimum (inclusive).
        :param max_value: The range maximum (exclusive).
        :param dist: The sampling distribution which can be 'uniform' or 'log10'.
        :return: The random number.
        '''
        if dist == 'uniform':
            return self.rng.uniform(min_value, max_value)
        if dist == 'log10':
            return 10**self.rng.uniform(math.log10(min_value), math.log10(max_value))
        raise ValueError('Distribution must be \'uniform\' or \'log10\'.')
