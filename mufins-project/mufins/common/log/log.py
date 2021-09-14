'''
Abstract log class for recording messages and presenting them to the user.
'''

from abc import ABC, abstractmethod
from typing import Optional
from mufins.common.log.progress_spec import ProgressSpec


#########################################
class Log(ABC):
    '''
    Abstract log class.
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Constructor.
        '''
        self.inited: bool = False

    #########################################
    @abstractmethod
    def init(
        self,
    ) -> bool:
        '''
        Create the log if it does not exist.

        :return: Whether the file was created or not.
        '''

    #########################################
    @abstractmethod
    def log_message(
        self,
        text: str = '',
    ) -> None:
        '''
        Log a line of message.

        :param text: The line of text.
        '''

    #########################################
    @abstractmethod
    def log_error(
        self,
        text: str,
    ) -> None:
        '''
        Log a line of error.

        :param text: The error message.
        '''

    #########################################
    @abstractmethod
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

    #########################################
    @abstractmethod
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

    #########################################
    @abstractmethod
    def progress_update(
        self,
        curr: int,
    ) -> None:
        '''
        Update the current progress section.

        :param curr: The number of the iteration that has completed.
        '''

    #########################################
    @abstractmethod
    def progress_end(
        self,
    ) -> None:
        '''
        Destroy the progress section and make the previous section the current one.
        '''
