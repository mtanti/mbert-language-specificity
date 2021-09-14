'''
Checkpoint handle module.
'''

from typing import Optional


#########################################
class SkipCheckpoint(Exception):
    '''Special exception for skipping the checkpoint with block.'''


#########################################
class CheckpointHandle():
    '''
    The object returned by the checkpoint context manager.
    '''

    #########################################
    def __init__(
        self,
        ready: bool,
        value: Optional[str],
    ) -> None:
        '''
        Constructor.

        :param ready: Whether the checkpoint was finished previously.
        :param value: The value associated with this checkpoint.
        '''
        self.ready: bool = ready
        self.value: Optional[str] = value

    #########################################
    def was_found_ready(
        self,
    ) -> bool:
        '''
        Get whether this checkpoint was already completed in a previous run.

        :return: Whether this checkpoint is ready.
        '''
        return self.ready

    #########################################
    def get_value(
        self,
    ) -> Optional[str]:
        '''
        Get the value associated with this checkpoint.

        :return: The value associated with this checkpoint.
        '''
        return self.value

    #########################################
    def set_value(
        self,
        new_value: Optional[str],
    ) -> None:
        '''
        Set the new value to associate with this checkpoint.

        :param new_value: The value associated with this checkpoint.
        '''
        self.value = new_value

    #########################################
    @staticmethod
    def skip(
    ) -> None:
        '''
        Skip this checkpoint without modification.

        Works by raising a special exception which is caught by the checkpoint
        manager.
        '''
        raise SkipCheckpoint()
