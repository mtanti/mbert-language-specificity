'''
Module for displaying the progress of a process.
'''

import os
from typing import Optional
from mufins.common.time.time_utils import Timer, get_readable_duration
from mufins.common.datastruct.duration_queue import DurationQueue
from mufins.common.error.invalid_state import InvalidStateException


#########################################
class CliProgressbar():
    '''
    Class for keeping track of progress and displaying information about it.
    '''

    #########################################
    def __init__(
        self,
        init_iter: int,
        final_iter: int,
        label_size: int = 0,
        dur_queue_size: Optional[int] = None,
        line_size: Optional[int] = None,
        parent_bar: Optional['CliProgressbar'] = None,
    ) -> None:
        '''
        Constructor.

        :param init_iter: The initial iteration number.
        :param final_iter: The final iteration number.
        :param label_size: The number of characters to reserve for the label at the front.
            If zero, then there will not be a label section.
        :param dur_queue_size: The number of iteration durations to record in
            a queue to use when measuring the average speed.
            If None then no individual durations are stored but only their
            total.
        :param line_size: The number of characters in a line.
        :param parent_bar: In case of nested bars, this would be a reference to the parent.
        '''
        self.curr_iter: int = init_iter
        self.final_iter: int = final_iter
        self.label_size: int = label_size
        self.dur_queue: DurationQueue = DurationQueue(dur_queue_size)
        self.line_size: Optional[int] = line_size
        self.parent_bar: Optional[CliProgressbar] = parent_bar
        self.label: str = ''
        self.final_iter_num_digits: int = len(str(self.final_iter))
        self.iter_timer: Timer = Timer()
        self.full_timer: Timer = Timer()

    #########################################
    def set_label(
        self,
        label: str,
    ) -> None:
        '''
        Update the label to show at the front of the progress bar.

        :param label: The text to show. Note that if this is longer than label_size then it will be
            shortened from the back.
        '''
        self.label = label

    #########################################
    def get_duration_queue(
        self,
    ) -> DurationQueue:
        '''
        Duration queue getter.

        :return: The duration queue.
        '''
        return self.dur_queue

    #########################################
    def set_duration_queue(
        self,
        dur_queue: DurationQueue,
    ) -> None:
        '''
        Duration queue setter.

        :param dur_queue: The duration queue.
        '''
        self.dur_queue = dur_queue

    #########################################
    def get_line_size(
        self,
    ) -> int:
        '''
        Get the width of a line in the CLI.

        :return: The width.
        '''
        if self.line_size is not None:
            return self.line_size
        return os.get_terminal_size().columns - 1

    #########################################
    def get_bar_str(
        self,
    ) -> str:
        '''
        Get the progress as a printable string.

        :return: The progress information as a printable string.
        '''
        perc = self.curr_iter/self.final_iter

        if self.dur_queue.get_count() > 0:
            avg_iter_duration = (
                (
                    self.dur_queue.get_total()
                    + self.iter_timer.get_duration()
                )/self.dur_queue.get_count()
            )
            time_left = (self.final_iter - self.curr_iter)*avg_iter_duration

            time_left_str = get_readable_duration(time_left)
            if avg_iter_duration > 9999.9:
                avg_iter_duration_str = ' '*6
            else:
                avg_iter_duration_str = '{: >6.1f}'.format(avg_iter_duration)
        else:
            time_left_str = '??s'
            avg_iter_duration_str = ' '*6

        time_elapsed_str = get_readable_duration(
            self.full_timer.get_duration()
        )

        line = '{}{: >4.0%} {: >{}d}/{: >{}d} {} ETA-{}@{}s/it'.format(
            '' if self.label_size == 0 else '[{: <{}s}]'.format(
                self.label[:self.label_size], self.label_size
            ),
            perc,
            self.curr_iter, self.final_iter_num_digits,
            self.final_iter, self.final_iter_num_digits,
            time_elapsed_str,
            time_left_str,
            avg_iter_duration_str,
        )

        if self.parent_bar is None:
            return line
        return '{} > {}'.format(self.parent_bar.get_bar_str(), line)

    #########################################
    def __refresh(
        self,
        clear: bool = False,
    ) -> None:
        '''
        Display a new progress line.

        :param clear: Whether to clear the line instead of show the progress.
        '''
        line_size = self.get_line_size()
        if not clear:
            line = self.get_bar_str()
            if len(line) > line_size:
                line = line[:line_size-3] + '...'
        else:
            line = ''
        end = ' '*(line_size - len(line)) + '\r'
        print(line, end=end)

    #########################################
    def start(
        self,
    ) -> None:
        '''
        Show the initial progress bar.
        '''
        self.full_timer.start()
        self.iter_timer.start()
        self.__refresh()

    #########################################
    def reset(
        self,
    ) -> None:
        '''
        Set the current iteration to zero and the label to empty.
        '''
        self.curr_iter = 0
        self.label = ''

    #########################################
    def update(
        self,
        new_curr_iter: int,
    ) -> None:
        '''
        Update the progress bar.

        :param new_curr_iter: The new current iteration.
        '''
        if self.full_timer.get_num_runs() == 0:
            raise InvalidStateException('Cannot call update() before calling start().')

        iterations = new_curr_iter - self.curr_iter
        if iterations > 0:
            iter_duration = self.iter_timer.get_duration()/iterations
            for _ in range(iterations):
                self.curr_iter = new_curr_iter
                self.dur_queue.add(iter_duration)
            self.iter_timer.reset()
            self.iter_timer.start()

        self.__refresh()

    #########################################
    def close(
        self,
    ) -> None:
        '''
        Close the progress bar.
        '''
        self.__refresh(clear=True)
        self.full_timer.reset()
        self.iter_timer.reset()
        if self.parent_bar is not None:
            self.parent_bar.update(self.parent_bar.curr_iter)
