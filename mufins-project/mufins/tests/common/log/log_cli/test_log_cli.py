'''
Unit test for LogCli class in Log CLI module.
'''

import unittest
import contextlib
import io
from mufins.common.log.log_cli import LogCli


#########################################
class TestLogCli(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the LogCli class.
        '''
        log = LogCli(
            line_size = 18
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with contextlib.redirect_stderr(stderr):
                self.assertFalse(
                    log.init()
                )
                log.log_message('test')
                log.log_message('test test test test test')
                log.log_error('error')
                log.log_error('error error error error error')

                spec = log.progress_start(
                    start = 0,
                    total = 3,
                    dur_queue_size = None,
                )
                log.progress_update(1)
                log.progress_update(2)
                log.progress_update(3)
                log.progress_end()

                log.log_message('test')

                log.progress_restart(0, spec)
                log.progress_update(1)
                log.progress_update(2)
                log.progress_update(3)
                log.progress_end()

                log.log_message('test')

                log.progress_restart(2, spec)
                log.progress_update(3)
                log.progress_end()

                log.log_message('test')

        self.assertEqual(
            stdout.getvalue(),
            '''\
test
test test test
test test
  0% 0/3 00s ET...\r 33% 1/3 00s ET...\r\
 67% 2/3 00s ET...\r100% 3/3 00s ET...\r\
                  \r\
test
  0% 0/3 00s ET...\r 33% 1/3 00s ET...\r\
 67% 2/3 00s ET...\r100% 3/3 00s ET...\r\
                  \r\
test
 67% 2/3 00s ET...\r100% 3/3 00s ET...\r\
                  \r\
test
'''
        )
        self.assertEqual(
            stderr.getvalue(),
            '''\
\033[91merror\033[0m
\033[91merror error error\033[0m
\033[91merror error\033[0m
'''
        )


#########################################
if __name__ == '__main__':
    unittest.main()
