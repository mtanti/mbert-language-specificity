'''
Unit test for Timer class in time utilities module.
'''

import unittest
import time
from mufins.common.time.time_utils import Timer
from mufins.common.error.invalid_state import InvalidStateException


#########################################
class TestTimer(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_exceptions(
        self,
    ) -> None:
        '''
        Test for exceptions.
        '''
        timer = Timer()

        # Cannot get the duration of a timer that never started.
        with self.assertRaises(InvalidStateException):
            timer.get_duration()

        # Cannot pause a paused timer.
        with self.assertRaises(InvalidStateException):
            timer.pause()

        # Cannot start an unpaused timer.
        timer.start()
        with self.assertRaises(InvalidStateException):
            timer.start()

    #########################################
    def test_start_pause(
        self,
    ) -> None:
        '''
        Test the use of explicit start and pause methods.
        '''
        timer = Timer()
        self.assertEqual(timer.get_num_runs(), 0)

        for _ in range(2):
            for runs in range(1, 2+1):
                timer.start()
                time.sleep(1)
                self.assertEqual(round(timer.get_duration()), runs)
                timer.pause()
                self.assertEqual(round(timer.get_duration()), runs)
                self.assertEqual(timer.get_num_runs(), runs)
            timer.reset()

    #########################################
    def test_with_block(
        self,
    ) -> None:
        '''
        Test the use of the with block.
        '''
        with Timer() as timer:
            time.sleep(1)
            self.assertEqual(round(timer.get_duration()), 1)
        self.assertEqual(round(timer.get_duration()), 1)
        self.assertEqual(timer.get_num_runs(), 1)

        timer = Timer()
        self.assertEqual(timer.get_num_runs(), 0)
        for _ in range(2):
            for runs in range(1, 2+1):
                with timer:
                    time.sleep(1)
                    self.assertEqual(round(timer.get_duration()), runs)
                self.assertEqual(round(timer.get_duration()), runs)
                self.assertEqual(timer.get_num_runs(), runs)
            timer.reset()


#########################################
if __name__ == '__main__':
    unittest.main()
