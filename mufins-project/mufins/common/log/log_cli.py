'''
Class for logging information in the command line.
'''

import os
import sys
import textwrap
from typing import Optional, List, cast
from mufins.common.log.log import Log
from mufins.common.log.cli_progressbar import CliProgressbar
from mufins.common.error.invalid_state import InvalidStateException
from mufins.common.log.progress_spec import ProgressSpec


#########################################
class LogCli(Log):
    '''
    Command line logger.
    '''

    #########################################
    def __init__(
        self,
        line_size: Optional[int] = None,
    ) -> None:
        '''
        Constructor.

        :param line_size: The maximum width of a line in the window in
            characters.
            If None then os.get_terminal_size().columns will be used.
        '''
        super().__init__()
        self.line_size: Optional[int] = line_size
        self.progressbars: List[CliProgressbar] = cast(List[CliProgressbar], [])

    #########################################
    def __get_line_size(
        self,
    ) -> int:
        '''
        Get the width of a line in the CLI.

        :return: The width.
        '''
        if self.line_size is not None:
            return self.line_size
        return os.get_terminal_size().columns

    #########################################
    def init(
        self,
    ) -> bool:
        '''
        Create the log if it does not exist.

        :return: Whether the file was created or not.
        '''
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
        if text == '':
            print()
        else:
            for line in textwrap.wrap(text, self.__get_line_size()):
                print(line)

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
        for line in textwrap.wrap(text, self.__get_line_size()):
            print('\033[91m'+line+'\033[0m', file=sys.stderr)

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
        self.progressbars.append(CliProgressbar(
            start,
            total,
            label_size = len(label) if label is not None else 0,
            line_size = self.line_size,
            dur_queue_size = dur_queue_size,
            parent_bar = self.progressbars[-1] if len(self.progressbars) > 0 else None,
        ))
        if label is not None:
            self.progressbars[-1].set_label(label)
        self.progressbars[-1].start()

        return ProgressSpec(
            total,
            self.progressbars[-1].get_duration_queue(),
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
        dur_queue = progress_spec.get_duration_queue()
        if dur_queue is None:
            raise ValueError('Progress specification must have a duration queue set.')
        self.progressbars.append(CliProgressbar(
            start,
            progress_spec.get_total(),
            label_size = len(label) if label is not None else 0,
            line_size = self.line_size,
            dur_queue_size = None,
            parent_bar = self.progressbars[-1] if len(self.progressbars) > 0 else None,
        ))
        if label is not None:
            self.progressbars[-1].set_label(label)
        self.progressbars[-1].set_duration_queue(dur_queue)
        self.progressbars[-1].start()

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
        if len(self.progressbars) == 0:
            raise InvalidStateException('Cannot update an unstarted progress.')

        self.progressbars[-1].update(curr)

    #########################################
    def progress_end(
        self,
    ) -> None:
        '''
        Destroy the progress section and make the previous section the current one.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised log.')
        if len(self.progressbars) == 0:
            raise InvalidStateException('Cannot update an unstarted progress.')
        self.progressbars.pop().close()
