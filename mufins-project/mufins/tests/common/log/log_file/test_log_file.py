'''
Unit test for LogFile class in Log CLI module.
'''

import os
import unittest
import tempfile
import time
from mufins.common.log.log_file import LogFile


#########################################
class TestLogFile(unittest.TestCase):
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
            log = LogFile(os.path.join(path, 'test.txt'))

            self.assertTrue(
                log.init()
            )
            self.assertFalse(
                log.init()
            )

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the LogFile class.
        '''
        with tempfile.TemporaryDirectory() as path:
            log = LogFile(os.path.join(path, 'test.txt'))

            log.init()
            log.log_message('test')
            log.log_message('test test test test test')
            log.log_error('error')
            log.log_error('error error error error error')
            log.progress_start(
                start = 0,
                total = 3,
                dur_queue_size = None,
            )
            time.sleep(0.1)
            log.progress_update(1)
            time.sleep(0.1)
            log.progress_update(2)
            time.sleep(0.1)
            log.progress_update(3)
            log.progress_end()
            log.log_message('test')

            with open(os.path.join(path, 'test.txt'), 'r', encoding='utf-8') as f:
                self.assertRegex(
                    f.read(),
                    '''\
\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\ttest
\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\ttest test test test test
\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\terror
\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\terror error error error error
\t\titeration \\(out of 3\\)\tduration \\(s\\)
\t\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\t1\t0\\.[01]\\d+
\t\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\t2\t0\\.[01]\\d+
\t\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\t3\t0\\.[01]\\d+
\\d{4}/\\d{2}/\\d{2} \\d{2}:\\d{2}:\\d{2} GMT\ttest
'''
        )


#########################################
if __name__ == '__main__':
    unittest.main()
