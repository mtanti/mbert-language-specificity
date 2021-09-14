'''
Unit test for Dataset class in dataset module.
'''

import unittest
import math
from mufins.common.dataset.dataset import Dataset
from mufins.common.dataset.dataset_memory import DatasetMemory
from mufins.tests.common.dataset.mock_data_spec import MockDataSpec
from mufins.common.random.random_number_generator import RandomNumberGenerator


#########################################
class TestDataset(unittest.TestCase):
    '''
    Unit test class.
    '''

    ########################################
    def test_split_nokey(
        self,
    ) -> None:
        '''
        Test the Dataset class when it is split with no stratification key.
        '''
        spec = MockDataSpec()

        dset = DatasetMemory[str](spec)
        dset.init(3)
        dset.load(as_readonly=False)

        dset.set_row(0, {'a': [0,1,2], 'b':[10]})
        dset.set_row(1, {'a': [3,4,5], 'b':[11]})
        dset.set_row(2, {'a': [6,7,8], 'b':[12]})

        def factory(
            name: str, # pylint: disable=unused-argument
            size: int,
        ) -> Dataset[str]:
            sub_dataset = DatasetMemory[str](spec)
            sub_dataset.init(size)
            sub_dataset.load(as_readonly=False)
            return sub_dataset

        with self.assertRaises(ValueError):
            dset.split(
                [('a', 0.6), ('b', 0.6)],
                factory, None, RandomNumberGenerator(0)
            )

        for fractions in [[2/3, 1/3], [1/2, 1/2]]:
            for key in [
                None,
                lambda row:row['b'].tolist()[0],
            ]:
                splits = dset.split(
                    [('x', fractions[0]), ('y', fractions[1])],
                    factory,
                    key,
                    RandomNumberGenerator(0),
                )

                self.assertEqual(
                    splits[0].size,
                    math.ceil(fractions[0]*dset.size),
                    'fractions = {}, key = {}'.format(
                        fractions, key
                    ),
                )
                self.assertEqual(
                    splits[1].size,
                    math.floor(fractions[1]*dset.size),
                    'fractions = {}, key = {}'.format(
                        fractions, key
                    ),
                )
                self.assertEqual(
                    {
                        (
                            tuple(x.get_field(i, 'a').tolist()),
                            tuple(x.get_field(i, 'b').tolist()),
                        )
                        for x in splits
                        for i in range(x.size)
                    },
                    {((0,1,2), (10,)), ((3,4,5), (11,)), ((6,7,8), (12,))},
                    'fractions = {}, key = {}'.format(
                        fractions, key
                    ),
                )

                for split in splits:
                    split.close()

        dset.close()

    ########################################
    def test_split_withkey(
        self,
    ) -> None:
        '''
        Test the Dataset class when it is split with a stratification key.
        '''
        spec = MockDataSpec()

        dset = DatasetMemory[str](spec)
        dset.init(10)
        dset.load(as_readonly=False)

        dset.set_row(0, {'a': [0,0,0], 'b':[0]})
        dset.set_row(1, {'a': [1,1,1], 'b':[0]})
        dset.set_row(2, {'a': [2,2,2], 'b':[1]})
        dset.set_row(3, {'a': [3,3,3], 'b':[1]})
        dset.set_row(4, {'a': [4,4,4], 'b':[2]})
        dset.set_row(5, {'a': [5,5,5], 'b':[2]})
        dset.set_row(6, {'a': [6,6,6], 'b':[3]})
        dset.set_row(7, {'a': [7,7,7], 'b':[3]})
        dset.set_row(8, {'a': [8,8,8], 'b':[4]})
        dset.set_row(9, {'a': [9,9,9], 'b':[4]})

        def factory(
            name: str, # pylint: disable=unused-argument
            size: int,
        ) -> Dataset[str]:
            sub_dataset = DatasetMemory[str](spec)
            sub_dataset.init(size)
            sub_dataset.load(as_readonly=False)
            return sub_dataset

        splits = dset.split(
            [('x', 0.5), ('y', 0.5)],
            factory,
            lambda row:row['b'].tolist()[0],
            RandomNumberGenerator(0),
        )

        self.assertEqual(
            {splits[0].get_field(i, 'b').tolist()[0] for i in range(splits[0].size)},
            {splits[1].get_field(i, 'b').tolist()[0] for i in range(splits[1].size)},
        )

        for split in splits:
            split.close()

        dset.close()

    ########################################
    def test_batch(
        self,
    ) -> None:
        '''
        Test the Dataset class when it is batched.
        '''
        spec = MockDataSpec()

        dset = DatasetMemory[str](spec)
        dset.init(3)
        dset.load(as_readonly=False)

        dset.set_row(0, {'a': [0,1,2], 'b':[10]})
        dset.set_row(1, {'a': [3,4,5], 'b':[11]})
        dset.set_row(2, {'a': [6,7,8], 'b':[12]})

        for batch_size in [1, 2, 3]:
            batches = list(dset.get_batches(batch_size))
            for (i, x) in enumerate(batches):
                self.assertEqual(
                    x['a'].tolist(),
                    [
                        dset.get_field(j, 'a').tolist()
                        for j in range(
                            i*batch_size,
                            min((i+1)*batch_size, dset.size)
                        )
                    ],
                    'batch_size = {}'.format(batch_size),
                )

        dset.close()

    ########################################
    def test_stochastic_batch(
        self,
    ) -> None:
        '''
        Test the Dataset class when it is stochastically batched.
        '''
        spec = MockDataSpec()

        dset = DatasetMemory[str](spec)
        dset.init(4)
        dset.load(as_readonly=False)

        dset.set_row(0, {'a': [0,1,2], 'b':[10]})
        dset.set_row(1, {'a': [3,4,5], 'b':[11]})
        dset.set_row(2, {'a': [6,7,8], 'b':[12]})
        dset.set_row(3, {'a': [9,0,1], 'b':[13]})

        for capped_size in [None, 2, 3, 4]:
            for batch_size in [1, 2]:
                batches = list(dset.get_stochastic_batches(
                    batch_size, RandomNumberGenerator(0),
                    capped_size=capped_size
                ))
                for x in batches[:-1]:
                    self.assertEqual(
                        x['a'].shape[0],
                        batch_size,
                        'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                    )
                self.assertLessEqual(
                    batches[-1]['a'].shape[0],
                    batch_size,
                    'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                )

                self.assertEqual(
                    max(x['a'].shape[0] for x in batches),
                    min(batch_size, dset.size if capped_size is None else capped_size),
                    'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                )
                self.assertEqual(
                    sum(x['a'].shape[0] for x in batches),
                    dset.size if capped_size is None else capped_size,
                    'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                )
                self.assertEqual(
                    sum(x['b'].shape[0] for x in batches),
                    dset.size if capped_size is None else capped_size,
                    'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                )
                if capped_size is None or capped_size >= dset.size:
                    self.assertEqual(
                        {
                            tuple(y)
                            for x in batches
                            for y in x['a'].tolist()
                        },
                        {(0,1,2), (3,4,5), (6,7,8), (9,0,1)},
                        'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                    )
                    self.assertEqual(
                        {
                            tuple(y)
                            for x in batches
                            for y in x['b'].tolist()
                        },
                        {(10,), (11,), (12,), (13,)},
                        'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                    )
                else:
                    self.assertEqual(
                        len({
                            tuple(y)
                            for x in batches
                            for y in x['a'].tolist()
                        }),
                        capped_size,
                        'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                    )
                    self.assertTrue(
                        {
                            tuple(y)
                            for x in batches
                            for y in x['a'].tolist()
                        } <= {(0,1,2), (3,4,5), (6,7,8), (9,0,1)},
                        'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                    )
                    self.assertEqual(
                        len({
                            tuple(y)
                            for x in batches
                            for y in x['b'].tolist()
                        }),
                        capped_size,
                        'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                    )
                    self.assertTrue(
                        {
                            tuple(y)
                            for x in batches
                            for y in x['b'].tolist()
                        } <= {(10,), (11,), (12,), (13,)},
                        'batch_size = {}, capped_size = {}'.format(batch_size, capped_size),
                    )

        dset.close()


#########################################
if __name__ == '__main__':
    unittest.main()
