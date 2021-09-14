'''
Class for composing several log objects together.
'''

from typing import Sequence, Optional
from mufins.common.log.log import Log
from mufins.common.error.invalid_state import InvalidStateException
from mufins.common.log.progress_spec import ProgressSpec


#########################################
class LogComposite(Log):
    '''
    Composite logger.
    '''

    #########################################
    def __init__(
        self,
        logs: Sequence[Log],
    ) -> None:
        '''
        Constructor.

        :param logs: A sequence of log objects to use together.
        '''
        super().__init__()
        self.logs: Sequence[Log] = logs

    #########################################
    def init(
        self,
    ) -> bool:
        '''
        Create the log if it does not exist.

        :return: Whether the file was created or not.
            In this case it tells you whether at least one of the init methods
            in the component logs returned True.
        '''
        result = False
        for log_ in self.logs:
            tmp = log_.init()
            result = result or tmp
        self.inited = True
        return result

    #########################################
    def log_message(
        self,
        text: str = '',
    ) -> None:
        '''
        Log a line of message.

        :param text: The line of text.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        for log_ in self.logs:
            log_.log_message(text)

    #########################################
    def log_error(
        self,
        text: str,
    ) -> None:
        '''
        Log a line of error.

        :param text: The error message.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        for log_ in self.logs:
            log_.log_error(text)

    #########################################
    def progress_start(
        self,
        start: int,
        total: int,
        label: Optional[str] = None,
        dur_queue_size: Optional[int] = None,
    ) -> ProgressSpec:
        '''
        Start a new progress information section and make it current.

        :param start: The starting iteration in the progress.
            Normally 0, but can be more if process is resumed.
        :param total: The total number of iterations in the process.
        :param label: A label for the current progress.
        :param dur_queue_size: The number of latest iteration durations to
            record in history to use when calculating the average duration
            (useful for when the average duration changes as iterations
            increase).
            If None then all the iteration durations are used to compute a
            global average duration.
        :return: The progress specification to be able to restart this progress.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        progress_spec = ProgressSpec(
            total,
            None,
        )
        for log_ in self.logs:
            progress_spec_ = log_.progress_start(start, total, label, dur_queue_size)
            if progress_spec_.get_duration_queue() is not None:
                progress_spec = progress_spec_
        return progress_spec

    #########################################
    def progress_restart(
        self,
        start: int,
        progress_spec: ProgressSpec,
        label: Optional[str] = None,
    ) -> None:
        '''
        Like progress_start but restarts a given progress section.

        :param start: The starting iteration in the progress.
            Normally 0, but can be more if process is resumed.
        :param progress_spec: The progress specification to restart.
        :param label: A label for the current progress.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        for log_ in self.logs:
            log_.progress_restart(start, progress_spec, label)

    #########################################
    def progress_update(
        self,
        curr: int,
    ) -> None:
        '''
        Update the current progress section.

        :param curr: The number of the iteration that has completed.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        for log_ in self.logs:
            log_.progress_update(curr)

    #########################################
    def progress_end(
        self,
    ) -> None:
        '''
        Destroy the progress section and make the previous section the current one.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        for log_ in self.logs:
            log_.progress_end()
