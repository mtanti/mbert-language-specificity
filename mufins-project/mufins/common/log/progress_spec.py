'''
Progress specification.
'''

from typing import Optional
from mufins.common.datastruct.duration_queue import DurationQueue


#########################################
class ProgressSpec():
    '''
    Class for storing information to be able to restart a progress section.
    '''

    #########################################
    def __init__(
        self,
        total: int,
        dur_queue: Optional[DurationQueue],
    ) -> None:
        '''
        Constructor.

        :param total: The total number of iterations in the process.
        :param dur_queue: The duration queue that stores information about iteration speed.
        '''
        self.total: int = total
        self.dur_queue: Optional[DurationQueue] = dur_queue

    #########################################
    def get_total(
        self,
    ) -> int:
        '''
        Total number of iterations getter.

        :return: The total number of iterations in the process.
        '''
        return self.total

    #########################################
    def get_duration_queue(
        self,
    ) -> Optional[DurationQueue]:
        '''
        Duration queue getter.

        :return: The duration queue.
        '''
        return self.dur_queue
