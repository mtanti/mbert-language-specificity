'''
Time utilities module.
'''

import datetime
import timeit
from types import TracebackType
from typing import Type, Optional
from mufins.common.error.invalid_state import InvalidStateException


#########################################
def get_readable_timestamp(
    timestamp: Optional[float] = None,
) -> str:
    '''
    Convert a UNIX timestamp into a readable string.

    :param timestamp: UNIX timestamp to convert. If None then current timestamp
        will be used.
    :return: The readable timestamp in the format 'YYYY/MM/DD-hh:mm:ss'.
    '''
    if timestamp is None:
        timestamp = datetime.datetime.now().timestamp()
    result = datetime.datetime.strftime(
        datetime.datetime.utcfromtimestamp(timestamp),
        '%Y/%m/%d %H:%M:%S GMT',
    )
    return result


#########################################
def get_readable_duration(
    duration_seconds: float,
) -> str:
    '''
    Convert a number of seconds into a readable string duration.

    :param duration_seconds: Number of seconds to convert.
    :return: The readable duration in the format '1d:01h:01m:01s' with larger
        units dropped if 0, but seconds unit always there.
    '''
    segs_in_secs = [
        24*60*60,
        60*60,
        60,
        1,
    ]
    dur = round(duration_seconds)
    segments = []
    for secs in segs_in_secs:
        (seg, dur) = divmod(dur, secs)
        segments.append(seg)
    assert dur == 0, 'Duration did not decrease to zero but to {}.'.format(dur)

    # This is to skip larger segments that are equal to zero.
    start = 0
    while start < len(segments) - 1 and segments[start] == 0:
        start += 1

    seg_formats = [
        '{:d}d',
        '{:0>2d}h',
        '{:0>2d}m',
        '{:0>2d}s',
    ]
    results = []
    for (i, (seg, fmt)) in enumerate(zip(segments, seg_formats)):
        if i < start:
            continue
        results.append(fmt.format(seg))
    return ':'.join(results)


#########################################
class Timer():
    '''
    A stopwatch type object for measuring durations.

    This object is meant to be used in a 'with' context, for example::

        with Timer() as t:
            #do something
        print(t.get_duration())  # Duration in seconds.

    It is also possible to use the same Timer object multiple times to get the
    total duration of all times, for example::

        t = Timer()
        with t:
            #do something
        with t:
            #do something else
        print(t.get_runs())  # 2 runs.
        print(t.get_duration())  # Total duration of both runs in seconds.
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Constructor.
        '''
        self._start: float = 0.0
        self.total_duration: float = 0.0
        self.runs: int = 0
        self.is_paused: bool = True

    #########################################
    def __enter__(
        self,
    ) -> 'Timer':
        '''
        Start the stopwatch.

        :return: This object.
        '''
        self.start()
        return self

    #########################################
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        '''
        Pause the stopwatch.

        :param exc_type: Exception type.
        :param exc_value: Exception value.
        :param traceback: Exception traceback.
        '''
        self.pause()

    #########################################
    def reset(
        self,
    ) -> None:
        '''
        Pause the stopwatch and reset it.
        '''
        self.total_duration = 0.0
        self.runs = 0
        self.is_paused = True

    #########################################
    def start(
        self,
    ) -> None:
        '''
        Start the stopwatch.
        '''
        if not self.is_paused:
            raise InvalidStateException('Cannot start an unpaused timer.')
        self._start = timeit.default_timer()
        self.runs += 1
        self.is_paused = False

    #########################################
    def pause(
        self,
    ) -> None:
        '''
        Pause the stopwatch.
        '''
        if self.is_paused:
            raise InvalidStateException('Cannot pause a paused timer.')
        self.total_duration += timeit.default_timer() - self._start
        self.is_paused = True

    #########################################
    def get_duration(
        self,
    ) -> float:
        '''
        Get the current duration in seconds. If the timer is running then the
            duration from first start to now is returned, otherwise the total
            unpaused duration is returned.

        :return: The current duration.
        '''
        if self.runs == 0:
            raise InvalidStateException(
                'Cannot get the duration of a timer that never started.'
            )

        if self.is_paused:
            return self.total_duration

        return self.total_duration + (timeit.default_timer() - self._start)

    #########################################
    def get_num_runs(
        self,
    ) -> int:
        '''
        Get the number of times the timer was started.

        :return: The number of times the timer was started.
        '''
        return self.runs
