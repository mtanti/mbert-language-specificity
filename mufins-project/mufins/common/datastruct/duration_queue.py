'''
Duration queue class.
'''

import collections
from typing import Deque, Optional


#########################################
class DurationQueue():
    '''
    A queue for storing the durations of iterations in a stream with the purpose
    of finding the average iteration duration.
    '''

    #########################################
    def __init__(
        self,
        size: Optional[int] = None,
    ) -> None:
        '''
        Constructor.

        :param size: The maximum number of iteration durations to record.
            If None then no individual durations are stored but only their
            total.
        '''
        self.size: Optional[int] = size
        self.queue: Deque[float] = collections.deque()
        self.total: float = 0.0
        self.count: int = 0

    #########################################
    def add(
        self,
        duration: float,
    ) -> None:
        '''
        Add another duration.

        :param duration: The next iteration's duration.
        '''
        if self.size is None:
            self.total += duration
            self.count += 1
        else:
            if len(self.queue) == self.size:
                self.total -= self.queue[0]
                self.queue.popleft()
            else:
                self.count += 1
            self.total += duration
            self.queue.append(duration)

    #########################################
    def get_total(
        self,
    ) -> float:
        '''
        Get the total recorded duration.

        :return: The total recorded duration.
        '''
        return self.total

    #########################################
    def get_count(
        self,
    ) -> int:
        '''
        Get the number of recorded durations.

        :return: The number of recorded durations.
        '''
        return self.count
