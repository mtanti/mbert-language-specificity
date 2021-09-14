'''
Unit test for DurationQueue class in duration queue module.
'''

import unittest
from mufins.common.datastruct.duration_queue import DurationQueue


#########################################
class TestDurationQueue(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_size_none(
        self,
    ) -> None:
        '''
        Test the DurationQueue class when it has unlimited size.
        '''
        queue = DurationQueue()

        self.assertEqual(queue.get_total(), 0.0)
        self.assertEqual(queue.get_count(), 0)

        total = 0.0
        for i in range(1, 1000 + 1):
            queue.add(float(i))
            total += float(i)
            self.assertEqual(queue.get_total(), total)
            self.assertEqual(queue.get_count(), i)

    #########################################
    def test_size_given(
        self,
    ) -> None:
        '''
        Test the DurationQueue class when it has a fixed size.
        '''
        queue = DurationQueue(
            size = 3,
        )

        self.assertEqual(queue.get_total(), 0.0)
        self.assertEqual(queue.get_count(), 0)

        queue.add(1.0)
        self.assertEqual(queue.get_total(), 1.0)
        self.assertEqual(queue.get_count(), 1)

        queue.add(2.0)
        self.assertEqual(queue.get_total(), 3.0)
        self.assertEqual(queue.get_count(), 2)

        queue.add(3.0)
        self.assertEqual(queue.get_total(), 6.0)
        self.assertEqual(queue.get_count(), 3)

        queue.add(4.0)
        self.assertEqual(queue.get_total(), 9.0)
        self.assertEqual(queue.get_count(), 3)

        queue.add(5.0)
        self.assertEqual(queue.get_total(), 12.0)
        self.assertEqual(queue.get_count(), 3)


#########################################
if __name__ == '__main__':
    unittest.main()
