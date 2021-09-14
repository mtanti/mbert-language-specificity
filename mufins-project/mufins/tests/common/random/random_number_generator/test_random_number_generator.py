'''
Unit test for Random number generator class in random module.
'''

import unittest
from mufins.common.random.random_number_generator import RandomNumberGenerator


#########################################
class TestRandomNumberGenerator(unittest.TestCase):
    '''
    Unit test class.
    '''

    ########################################
    def test_seed(
        self,
    ) -> None:
        '''
        Test the seed management feature.
        '''
        rng = RandomNumberGenerator()
        self.assertTrue(isinstance(rng.seed, int))

        rng2 = RandomNumberGenerator()
        self.assertNotEqual(rng.seed, rng2.seed) # Very probable.

        rng = RandomNumberGenerator(0)
        self.assertEqual(rng.seed, 0)

    ########################################
    def test_get_child(
        self,
    ) -> None:
        '''
        Test the get_child method.
        '''
        rng = RandomNumberGenerator(0)
        rng2 = rng.get_child()
        rng3 = rng.get_child()
        self.assertNotEqual(rng2.seed, 0)
        self.assertNotEqual(rng2.seed, rng3.seed)

        rng = RandomNumberGenerator(0)
        rng3 = rng.get_child()
        self.assertEqual(rng2.seed, rng3.seed)

    ########################################
    def test_shuffle(
        self,
    ) -> None:
        '''
        Test the shuffle method.
        '''
        rng = RandomNumberGenerator(0)
        x = [1, 2, 3, 4]
        rng.shuffle(x)
        self.assertNotEqual(x, [1, 2, 3, 4])
        y = list(x)
        rng.shuffle(y)
        self.assertNotEqual(x, y)

        rng = RandomNumberGenerator(0)
        y = [1, 2, 3, 4]
        rng.shuffle(y)
        self.assertEqual(x, y)

    ########################################
    def test_array_normal(
        self,
    ) -> None:
        '''
        Test the array_normal method.
        '''
        rng = RandomNumberGenerator(0)
        x = rng.array_normal(0.0, 1.0, (2, 3))
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(str(x.dtype), 'float32')
        y = rng.array_normal(0.0, 1.0, (2, 3))
        self.assertNotEqual(x.tolist(), y.tolist())

        rng = RandomNumberGenerator(0)
        y = rng.array_normal(0.0, 1.0, (2, 3))
        self.assertEqual(x.tolist(), y.tolist())


#########################################
if __name__ == '__main__':
    unittest.main()
