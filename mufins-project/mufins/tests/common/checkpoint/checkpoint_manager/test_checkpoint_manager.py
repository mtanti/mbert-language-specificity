'''
Unit test for Checkpoint Manager class in checkpoint module.
'''

import os
import unittest
import tempfile
from mufins.common.checkpoint.checkpoint_manager import CheckpointManager
from mufins.common.error.incompatible_existing_data import IncompatibleExistingDataException


#########################################
class TestCheckpointManager(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_init(
        self,
    ) -> None:
        '''
        Test the init method.
        '''
        with tempfile.TemporaryDirectory() as path:
            ckpt_mgr = CheckpointManager(os.path.join(path, 'checkpoint.sqlite3'))

            self.assertTrue(
                ckpt_mgr.init()
            )
            self.assertFalse(
                ckpt_mgr.init()
            )

            with open(os.path.join(path, 'checkpoint.sqlite3'), 'w') as f:
                print('x', file=f)
            with self.assertRaises(IncompatibleExistingDataException):
                ckpt_mgr.init()

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the CheckpointManager class.
        '''
        # pylint: disable=redundant-unittest-assert
        with tempfile.TemporaryDirectory() as path:
            ckpt_mgr = CheckpointManager(os.path.join(path, 'checkpoint.sqlite3'))
            ckpt_mgr.init()

            with ckpt_mgr.checkpoint('c1') as hnd:
                self.assertFalse(hnd.was_found_ready())
                hnd.set_value('c1')

            with ckpt_mgr.checkpoint('c2') as hnd:
                self.assertFalse(hnd.was_found_ready())

            with ckpt_mgr.checkpoint('c1') as hnd:
                self.assertTrue(hnd.was_found_ready())
                self.assertEqual(hnd.get_value(), 'c1')
                hnd.set_value('x1')

            with ckpt_mgr.checkpoint('c1') as hnd:
                self.assertTrue(hnd.was_found_ready())
                self.assertEqual(hnd.get_value(), 'x1')
                hnd.skip()
                self.assertTrue(False)

            with ckpt_mgr.checkpoint('c1') as hnd:
                self.assertTrue(hnd.was_found_ready())
                self.assertEqual(hnd.get_value(), 'x1')
                hnd.set_value('y1')
                hnd.skip()
                self.assertTrue(False)

            with ckpt_mgr.checkpoint('c2') as hnd:
                self.assertTrue(hnd.was_found_ready())
                self.assertEqual(hnd.get_value(), None)
                hnd.skip()
                self.assertTrue(False)

            ckpt_mgr = CheckpointManager(os.path.join(path, 'checkpoint.sqlite3'))
            ckpt_mgr.init()

            with ckpt_mgr.checkpoint('c1') as hnd:
                self.assertTrue(hnd.was_found_ready())
                self.assertEqual(hnd.get_value(), 'y1')
                hnd.skip()
                self.assertTrue(False)

            with ckpt_mgr.checkpoint('c2') as hnd:
                self.assertTrue(hnd.was_found_ready())
                self.assertEqual(hnd.get_value(), None)
                hnd.skip()
                self.assertTrue(False)


#########################################
if __name__ == '__main__':
    unittest.main()
