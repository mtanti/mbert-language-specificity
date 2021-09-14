'''
Class for logging information in a file.
'''

from typing import Optional, List, cast
from mufins.common.log.log import Log
from mufins.common.time.time_utils import Timer, get_readable_timestamp
from mufins.common.error.invalid_state import InvalidStateException
from mufins.common.log.progress_spec import ProgressSpec


#########################################
class LogFile(Log):
    '''
    Text file logger.
    '''

    #########################################
    def __init__(
        self,
        path: str,
        show_progress: bool = True,
    ) -> None:
        '''
        Constructor.

        :param path: The path where to save the log file.
        :param show_progress: Whether to save progress in the file.
        '''
        super().__init__()
        self.path: str = path
        self.show_progress: bool = show_progress
        self.iter_timers: List[Timer] = cast(List[Timer], [])

    #########################################
    def init(
        self,
    ) -> bool:
        '''
        Create the log if it does not exist.

        :return: Whether the file was created or not.
        '''
        try:
            with open(self.path, 'x', encoding='utf-8'):
                self.inited = True
                return True
        except FileExistsError:
            self.inited = True
            return False

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
        with open(self.path, 'a', encoding='utf-8') as f:
            print(get_readable_timestamp(), text, sep='\t', file=f)

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
        with open(self.path, 'a', encoding='utf-8') as f:
            print(get_readable_timestamp(), text, sep='\t', file=f)

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
        if self.show_progress:
            self.iter_timers.append(Timer())
            self.iter_timers[-1].start()
            with open(self.path, 'a', encoding='utf-8') as f:
                print(
                    '\t'*((len(self.iter_timers) - 1)*3),
                    '' if label is None else label,
                    'iteration (out of {})'.format(total),
                    'duration (s)',
                    sep='\t',
                    file=f,
                )
        return ProgressSpec(
            total,
            None,
        )

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
        if self.show_progress:
            self.iter_timers.append(Timer())
            self.iter_timers[-1].start()
            with open(self.path, 'a', encoding='utf-8') as f:
                print(
                    '\t'*((len(self.iter_timers) - 1)*3),
                    '' if label is None else label,
                    'iteration (out of {})'.format(progress_spec.get_total()),
                    'duration (s)',
                    sep='\t',
                    file=f,
                )

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
        if self.show_progress:
            if len(self.iter_timers) == 0:
                raise InvalidStateException('Cannot update an unstarted progress.')
            with open(self.path, 'a', encoding='utf-8') as f:
                print(
                    '\t'*((len(self.iter_timers) - 1)*3),
                    get_readable_timestamp(),
                    curr,
                    self.iter_timers[-1].get_duration(),
                    sep='\t',
                    file=f,
                )
            self.iter_timers[-1].reset()
            self.iter_timers[-1].start()

    #########################################
    def progress_end(
        self,
    ) -> None:
        '''
        Destroy the progress section and make the previous section the current one.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        if self.show_progress:
            if len(self.iter_timers) == 0:
                raise InvalidStateException('Cannot update an unstarted progress.')
            self.iter_timers.pop().reset()
