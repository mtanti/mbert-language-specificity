'''
Abstract dataset class for storing and using datasets.
'''

import math
import collections
import itertools
import typing
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import (
    Callable, Hashable, Generic, Iterator, Mapping, Sequence, TypeVar,
    Union, List, Tuple, Optional, cast
)
import h5py
import numpy as np
from mufins.common.random.random_number_generator import RandomNumberGenerator
from mufins.common.dataset.data_spec import DataSpec
from mufins.common.error.invalid_state import InvalidStateException
from mufins.common.log.log import Log


#########################################
def get_batch_size(
    batch: Mapping[str, np.ndarray],
) -> int:
    '''
    Get the number of rows in a batch of rows.

    :param batch: The batch.
    :return: The size.
    '''
    try:
        return batch[next(iter(batch))].shape[0]
    except StopIteration:
        raise ValueError('Batch has no fields.') from None


#########################################
def concatenate_subbatches(
    subbatches: Sequence[Mapping[str, np.ndarray]],
) -> Mapping[str, np.ndarray]:
    '''
    Concatenate a sequence of single row subbatches together into a single batch.

    The keys of the first subbatch are used for the large batch.

    :param subbatches: The sequence of single row subbatches.
    :return: The single batch.
    '''
    keys = next(iter(subbatches)).keys()
    return {
        key: np.stack([
            subbatch[key]
            for subbatch in subbatches
        ])
        for key in keys
    }


#########################################
def get_subbatches(
    batch: Mapping[str, np.ndarray],
    subbatch_size: int,
) -> Iterator[Mapping[str, np.ndarray]]:
    '''
    Get batches of rows from a larger batch in its current order.

    The batches will all be of the same size, except for the
    last batch which will have the remainder.

    :param batch: The monolitic batch to break down.
    :param subbatch_size: The number of rows in each subbatch.
    :return: An iterator of batches.
    '''
    size = get_batch_size(batch)

    if size <= subbatch_size:
        yield batch
    else:
        for i in range(math.ceil(size/subbatch_size)):
            yield {
                name: batch[name][i*subbatch_size:(i+1)*subbatch_size]
                for name in batch
            }


#########################################
T = TypeVar('T')
MapT = TypeVar('MapT')

class Dataset(ABC, Generic[T]):
    '''
    Dataset class.

    T is a class for composite objects of raw data (such as texts or images).
    This object is expected to contain all the information necessary to
    produce an entire line of fields in the dataset, including target values.

    MapT is a class for the value mapper return type in get_data.
    '''

    #########################################
    def __init__(
        self,
        spec: DataSpec[T],
    ) -> None:
        '''
        Constructor.

        :param spec: The data specification object that describes the data
            to store in the dataset.
        '''
        self.spec: DataSpec[T] = spec
        self.data: Union[h5py.File, Mapping[str, np.ndarray], None] = None
        self.size: int = 0
        self.readonly: bool = False

    #########################################
    @abstractmethod
    def init(
        self,
        size: Optional[int] = None,
    ) -> bool:
        '''
        Prepare to load the dataset and create it if it does not exist.

        :param size: The number of rows to reserve in the dataset.
            If None then the data is expected to already exist.
        :return: Whether the file was created or not.
        '''
        # Process should go as follows:
        # if data already exists:
        #     load data
        #     if size is None:
        #         size = self.size
        #     elif self.size != size:
        #         raise error
        #     check data
        #     return False
        # else:
        #     if size is None:
        #         raise ValueError
        #     if self.size > 0 and size != self.size:
        #         raise error
        #     create data
        #     self.size = size
        #     return True if a file was created else False

    #########################################
    @abstractmethod
    def load(
        self,
        as_readonly: bool = False,
    ) -> None:
        '''
        Load the dataset.

        :param as_readonly: Whether the dataset can be modified or not.
            Only effective for saved datasets.
        '''

    #########################################
    def get_row(
        self,
        index: int,
    ) -> Mapping[str, np.ndarray]:
        '''
        Get a row from the dataset.

        :param index: The 0-based row number to get.
        :return: A mapping of field names to numpy arrays of field values.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to get rows from.')

        return {
            name: self.data[name][index]
            for name in self.data
        }

    #########################################
    def get_field(
        self,
        index: int,
        name: str,
    ) -> np.ndarray:
        '''
        Get a field from row in the dataset.

        :param index: The 0-based row number to get.
        :param name: The field name to get.
        :return: The numpy array with the field value.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to get rows from.')

        return self.data[name][index]

    #########################################
    def set_row(
        self,
        index: int,
        data: Mapping[str, np.ndarray],
    ) -> None:
        '''
        Set a row in the dataset.

        :param index: The 0-based row number to set.
        :param data: A mapping of field names to numpy arrays to set the row to.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to set rows in.')
        if self.readonly:
            raise InvalidStateException('Cannot change a readonly dataset.')

        for name in self.data:
            self.data[name][index] = data[name]

    #########################################
    def set_field(
        self,
        index: int,
        name: str,
        data: np.ndarray,
    ) -> None:
        '''
        Set a field in a row in the dataset.

        :param index: The 0-based row number to set.
        :param name: The field name to set.
        :param data: A numpy array to set the field to.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to set rows in.')
        if self.readonly:
            raise InvalidStateException('Cannot change a readonly dataset.')

        self.data[name][index] = data

    #########################################
    def get_num_batches(
        self,
        batch_size: int,
        whole_size: Optional[int] = None,
    ) -> int:
        '''
        Get the number of batches that will be formed from this dataset.

        :param batch_size: The number of rows in each batch.
        :param whole_size: The size of the data set.
            If None then the size of this data set will be used.
        :return: The number of batches.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to get number of batches from.')

        if whole_size is None:
            whole_size = self.size

        return math.ceil(whole_size/batch_size)

    #########################################
    def get_batches(
        self,
        batch_size: int,
    ) -> Iterator[Mapping[str, np.ndarray]]:
        '''
        Get batches of rows from the dataset in their current order.

        The batches will all be of the same size, except for the
        last batch which will have the remainder.

        :param batch_size: The number of rows in each batch.
        :return: An iterator of batches.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to get batches from.')

        for i in range(self.get_num_batches(batch_size)):
            slc = slice(i*batch_size, (i+1)*batch_size)
            yield {
                name: self.data[name][slc]
                for name in self.data
            }

    #########################################
    def get_stochastic_batches(
        self,
        batch_size: int,
        rng: RandomNumberGenerator,
        capped_size: Optional[int] = None,
    ) -> Iterator[Mapping[str, np.ndarray]]:
        '''
        Get random batches of rows from the dataset.

        The batches will all be unique and of the same size, except for the
        last batch which will have the remainder.

        :param batch_size: The number of rows in each batch.
        :param rng: The random number generator to use.
        :param capped_size: The maximum number of rows to take from the
            dataset in total. If data set size is less than the cap then the data set will be
            recycled through, each time in a new shuffled order, until the total number of items
            in all the batches returned is equal to the cap.
            If None then full dataset is used.
        :return: An iterator of batches.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to get batches from.')
        if batch_size*2 > self.size:
            raise ValueError(
                'Batch size is too big. Data set must be at least twice as large as the batch size.'
            )

        if capped_size is None:
            capped_size = self.size

        def get_indexes(
            rng: RandomNumberGenerator,
        ) -> Iterator[int]:
            indexes = list(range(self.size))
            rng.shuffle(indexes)
            i = 0
            while True:
                yield indexes[i]
                i += 1
                if i == len(indexes):
                    indexes_ = list(indexes)
                    while True:
                        rng.shuffle(indexes)
                        if len(set(indexes[:batch_size]) & set(indexes_[-batch_size:])) == 0:
                            break
                    i = 0

        total_returned = 0
        num_batches = self.get_num_batches(batch_size, capped_size)
        indexes = get_indexes(rng.get_child())
        for i in range(1, num_batches + 1):
            batch_indexes = list(itertools.islice(indexes, batch_size))

            total_returned += len(batch_indexes)
            if total_returned > capped_size:
                assert i == num_batches
                batch_indexes = batch_indexes[:-(total_returned - capped_size)]

            batch_indexes.sort()
            yield {
                name: self.data[name][batch_indexes]
                for name in self.data
            }


    #########################################
    def get_data(
        self,
        batch_size: int,
        value_filter: Optional[Callable[[int, Mapping[str, np.ndarray]], bool]] = None,
        value_mapper: Optional[Callable[[int, Mapping[str, np.ndarray]], MapT]] = None,
    ) -> Iterator[MapT]:
        '''
        Get the entire data from the dataset.

        :param batch_size: The number of rows in each batch.
        :param value_filter: A function to tell whether to include a given row into the resultant
            array.
            Signature of function is filter(index, row) -> include_item.
            If None then all rows are included.
        :param value_mapper: A function to transform a single row of data before including it in
            the resultant array.
            Signature of function is map(index, row) -> new_item.
            If None then rows themselves are added to array.
        :return: A list of data values.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to get rows from.')

        if value_filter is None:
            value_filter = lambda index, row: True
        if value_mapper is None:
            value_mapper = lambda index, row: cast(MapT, row)

        for (i, batch) in enumerate(self.get_batches(batch_size)):
            for j in range(get_batch_size(batch)):
                index = i*batch_size + j
                row = {name: batch[name][j] for name in batch}
                if value_filter(index, row):
                    yield value_mapper(index, row)

    #########################################
    def _create_splits(
        self,
        indexes: List[int],
        partition_fractions: Sequence[float],
        stratification_key: Union[
            Callable[[Mapping[str, np.ndarray]], Hashable],
            None
        ],
        rng: RandomNumberGenerator,
    ) -> Sequence[Sequence[int]]:
        '''
        Get the row indexes to use in each split.

        :param indexes: The indexes of all rows in the dataset, shuffled.
        :param partition_fractions: As described in split.
        :param stratification_key: As described in split.
        :param rng: As described in split.
        :return: A list of row indexes for each requested split.
        '''
        rng.shuffle(indexes)

        if stratification_key is None:
            strata = typing.cast(
                Mapping[Hashable, List[int]],
                {None: indexes}
            )
        else:
            strata = typing.cast(
                Mapping[Hashable, List[int]],
                collections.defaultdict(list)
            )
            for index in indexes:
                strata[stratification_key(self.get_row(index))].append(index)

        splits = typing.cast(
            List[List[int]],
            [[] for _ in range(len(partition_fractions))]
        )
        residue = []
        for stratum in strata.values():
            split_sizes = [
                int(len(stratum)*fraction)
                for fraction in partition_fractions
            ]
            assert len(stratum) - sum(split_sizes) <= len(splits)

            # If the stratum contains a number of items which is not evenly
            # divisible by the split fractions then move the remainder from the
            # end of the stratum to a residue list.
            if len(stratum) != sum(split_sizes):
                residue.extend(stratum[-(len(stratum) - sum(split_sizes)):])

            start = 0
            for (split, size) in zip(splits, split_sizes):
                split.extend(stratum[start:(start + size)])
                start += size

        # Distribute the residue as if it is a single stratum.
        if len(residue) > 0:
            split_sizes = [
                int(len(residue)*fraction)
                for fraction in partition_fractions
            ]
            for i in range(len(residue) - sum(split_sizes)):
                split_sizes[i] += 1

            start = 0
            rng.shuffle(residue)
            for (split, size) in zip(splits, split_sizes):
                split.extend(residue[start:(start + size)])
                start += size

        return splits

    #########################################
    def split(
        self,
        partition_fractions: Sequence[Tuple[str, float]],
        dataset_factory: Callable[[str, int], 'Dataset'],
        stratification_key: Union[
            Callable[[Mapping[str, np.ndarray]], Hashable],
            None
        ],
        rng: RandomNumberGenerator,
        log: Optional[Log] = None,
    ) -> Sequence['Dataset[T]']:
        '''
        Split a big dataset into smaller datasets.

        The splits are specified as a list of fractions. If the number of rows
        is not a multiple of the number of splits, then there will be some
        splits which have one more row than they should. The order of these splits
        should indicate a priority such that the first split will always have
        the extra row in this situation whilst the last split will be the last
        to have it.

        :param partition_fractions: A sequence of dataset name / fraction pairs.
            The names are passed to the dataset_factory.
            The fractions are the fractions of rows to take for each sub dataset
            and must sum to 1.0.
        :param dataset_factory: A function that returns empty subdatasets for
            the splits. The returned dataset must be initialised and loaded as
            writable. The signature for the functions is:
            dataset_factory(name, size) -> empty subdataset
        :param stratification_key: In order to balance the kind of data in all
            the sub-datasets, the rows can be grouped into strata and each
            stratum be split equally among the sub-datasets. The
            stratification_key function is supposed to take a row from the
            original dataset (given as a parameter) and return a hashable key.
            Rows with the same key are put in the same stratum.
        :param rng: The random number generator to use.
        :param log: The log to pass progress information to.
        :return: A sequence of datasets, one for each split. Datasets will not
            be closed.
        '''
        if self.data is None:
            raise InvalidStateException('No data loaded to get batches from.')
        if round(sum(f for (_, f) in partition_fractions), 7) != 1.0:
            raise ValueError('Fractions in partition_fractions must sum to 1.0.')
        for (_, fraction) in partition_fractions:
            if int(self.size*fraction) == 0:
                raise ValueError(
                    'Dataset is too small to be split into the given '
                    + 'fractions as some splits will end up empty.'
                )

        indexes = list(range(self.size))

        splits = self._create_splits(
            indexes,
            [fraction for (_, fraction) in partition_fractions],
            stratification_key,
            rng,
        )

        partition_names = [name for (name, _) in partition_fractions]
        sub_datasets = typing.cast('List[Dataset[T]]', [])
        if log is not None:
            log.progress_start(0, sum(len(x) for x in splits))
        i = 0
        for (name, split) in zip(partition_names, splits):
            sub_dataset = dataset_factory(name, len(split))
            for (dst_index, src_index) in enumerate(split):
                sub_dataset.set_row(dst_index, self.get_row(src_index))
                i += 1
                if log is not None:
                    log.progress_update(i)
            sub_datasets.append(sub_dataset)
        if log is not None:
            log.progress_end()
        return sub_datasets

    #########################################
    @abstractmethod
    def close(
        self,
    ) -> None:
        '''
        Close the dataset.
        '''


#########################################
@contextmanager
def autocloser(
    datasets: Sequence[Dataset[T]],
) -> Iterator[None]:
    '''
    A context manager to automatically close multiple data sets.

    :param datasets: The sequence of data sets to close.
    :return: Nothing.
    '''
    try:
        yield None
    finally:
        for dset in datasets:
            dset.close()
