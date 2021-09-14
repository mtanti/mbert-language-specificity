'''
Unit test for LogComposite class in Log Composite module.
'''

import os
import unittest
import tempfile
import time
from mufins.common.log.log_file import LogFile
from mufins.common.log.log_composite import LogComposite


#########################################
class TestLogComposite(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the LogComposite class.
        '''
        with tempfile.TemporaryDirectory() as path:
            fname1 = os.path.join(path, 'test1.txt')
            log1 = LogFile(
                path = fname1,
            )

            fname2 = os.path.join(path, 'test2.txt')
            log2 = LogFile(
                path = fname2,
            )

            log = LogComposite([log1, log2])

            self.assertTrue(
                log.init()
            )
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

            for fname in [fname1, fname2]:
                with open(fname, 'r', encoding='utf-8') as f:
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
