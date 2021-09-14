'''
Checkpoint module.
'''

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator
import shove
from mufins.common.checkpoint.checkpoint_handle import CheckpointHandle, SkipCheckpoint
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException
from mufins.common.error.invalid_state import InvalidStateException


#########################################
class CheckpointManager():
    '''
    A checkpoint object for skipping finished tasks.

    This object is meant to be used in a 'with' context, for example::

        ckpt_mgr = CheckpointManager('ckpt_file')
        ckpt_mgr.init()

        with ckpt_mgr.checkpoint('ckpt1') as handle:
            if handle.was_found_ready():
                # Checkpoint ckpt1 was already completed before so skip it.

                # Get the previous string value associated with this checkpoint.
                print(handle.get_value())

                handle.skip()

            # Do something.

            # Associate a string value with this checkpoint.
            handle.set_value('ready')

        # Checkpoint manager ckpt_mgr has recorded that ckpt1 was completed.
        # The value 'ready' is now stored for next run.
    '''

    #########################################
    def __init__(
        self,
        path: str,
    ) -> None:
        '''
        Constructor.

        :param path: The path to the checkpoint file.
            The file saved is a sqlite3 file.
        '''
        self.path: str = path
        self.inited: bool = False

    #########################################
    def init(
        self,
    ) -> bool:
        '''
        Create the checkpoint file if it does not exist.

        :return: Whether the file was created or not.
        '''
        # Even if file exists, still open it to check that it is valid.
        created = not os.path.isfile(self.path)
        try:
            shove.Shove('lite://'+self.path).close()
            self.inited = True
            return created
        except sqlite3.DatabaseError as ex:
            if str(ex) == 'file is not a database':
                raise IncompatibleExistingDataException('Existing file is incompatible.') \
                    from None
            raise ex

    #########################################
    @contextmanager
    def checkpoint(
        self,
        checkpoint_name: str,
    ) -> Iterator[CheckpointHandle]:
        '''
        :param checkpoint_name: A unique name for this checkpoint.
        :return: The checkpoint handle to use with this checkpoint.
        '''
        if not self.inited:
            raise InvalidStateException('Cannot use an uninitialised file.')
        was_ready = False
        value = None
        f = shove.Shove('lite://'+self.path)
        try:
            value = f[checkpoint_name]
            was_ready = True
        except KeyError:
            pass
        finally:
            f.close()

        handle = CheckpointHandle(was_ready, value)

        try:
            yield handle
        except SkipCheckpoint:
            pass

        f = shove.Shove('lite://'+self.path)
        try:
            f[checkpoint_name] = handle.get_value()
        finally:
            f.close()
