'''
Unit test for CliProgressBar class in CLI progressbar module.
'''

import unittest
import contextlib
import io
import time
import re
from mufins.common.log.cli_progressbar import CliProgressbar


#########################################
def wiggle(
    exact: str,
) -> str:
    '''
    Transform an expected progress information string into a regular expression
    with wiggle room for the speed to be slightly more or less.

    :param exact: The exact progress information expected if no time errors happened.
    :return: The regular expression.
    '''
    output = []
    tail = exact
    while True:
        match = re.search('[ 0-9]{4}\\.[0-9]', tail)
        if match is None:
            output.append(tail.replace('?', '\\?').replace('[', '\\['))
            break

        (start, end) = match.span()

        prefix = tail[:start]
        output.append(
            prefix.replace('?', '\\?').replace('[', '\\[')
        )

        speed = float(tail[start:end])
        if speed > 0.0:
            output.append(
                '({: >6.1f}|{: >6.1f}|{: >6.1f})'.format(
                    speed-0.1, speed, speed+0.1
                ).replace('.', '\\.')
            )
        else:
            output.append(
                '({: >6.1f}|{: >6.1f})'.format(
                    speed, speed+0.1
                ).replace('.', '\\.')
            )

        tail = tail[end:]

    return ''.join(output)


#########################################
class TestCliProgressbar(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_queue_size_none(
        self,
    ) -> None:
        '''
        Test when dur_queue_size is None.
        '''
        pbar = CliProgressbar(
            init_iter = 0,
            final_iter = 10,
            line_size = 33,
            dur_queue_size = None,
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.start()
        self.assertRegex(
            s.getvalue(),
            wiggle('  0%  0/10 00s ETA-??s@      s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar.update(1)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 10%  1/10 01s ETA-09s@   1.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar.update(6)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 60%  6/10 02s ETA-01s@   0.3s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar.update(10)
        self.assertRegex(
            s.getvalue(),
            wiggle('100% 10/10 03s ETA-00s@   0.3s/it\r')
        )

    #########################################
    def test_queue_size_set(
        self,
    ) -> None:
        '''
        Test when dur_queue_size is not None.
        '''
        pbar = CliProgressbar(
            init_iter = 0,
            final_iter = 10,
            line_size = 33,
            dur_queue_size = 2,
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.start()
        self.assertRegex(
            s.getvalue(),
            wiggle('  0%  0/10 00s ETA-??s@      s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar.update(1)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 10%  1/10 01s ETA-09s@   1.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(0.5)
            pbar.update(2)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 20%  2/10 02s ETA-06s@   0.8s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(0.5)
            pbar.update(3)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 30%  3/10 02s ETA-04s@   0.5s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(0.1)
            pbar.update(10)
        self.assertRegex(
            s.getvalue(),
            wiggle('100% 10/10 02s ETA-00s@   0.0s/it\r')
        )

    #########################################
    def test_zero_iterations(
        self,
    ) -> None:
        '''
        Test when an update is called with iterations is zero.
        '''
        pbar = CliProgressbar(
            init_iter = 0,
            final_iter = 3,
            line_size = 31,
            dur_queue_size = None,
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.start()
        self.assertRegex(
            s.getvalue(),
            wiggle('  0% 0/3 00s ETA-??s@      s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar.update(0)
        self.assertRegex(
            s.getvalue(),
            wiggle('  0% 0/3 01s ETA-??s@      s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar.update(1)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 33% 1/3 02s ETA-04s@   2.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar.update(1)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 33% 1/3 03s ETA-06s@   3.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.update(3)
        self.assertRegex(
            s.getvalue(),
            wiggle('100% 3/3 03s ETA-00s@   1.0s/it\r')
        )

    #########################################
    def test_label(
        self,
    ) -> None:
        '''
        Test when a label is used.
        '''
        pbar = CliProgressbar(
            init_iter = 0,
            final_iter = 3,
            label_size = 5,
            line_size = 38,
            dur_queue_size = None,
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.start()
        self.assertRegex(
            s.getvalue(),
            wiggle('[     ]  0% 0/3 00s ETA-??s@      s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.set_label('abc')
            pbar.update(1)
        self.assertRegex(
            s.getvalue(),
            wiggle('[abc  ] 33% 1/3 00s ETA-00s@   0.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.set_label('12345678')
            pbar.update(2)
        self.assertRegex(
            s.getvalue(),
            wiggle('[12345] 67% 2/3 00s ETA-00s@   0.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar.set_label('x')
            pbar.update(3)
        self.assertRegex(
            s.getvalue(),
            wiggle('[x    ]100% 3/3 00s ETA-00s@   0.0s/it\r')
        )

    #########################################
    def test_nesting(
        self,
    ) -> None:
        '''
        Test when a bar has a parent bar.
        '''
        pbar1 = CliProgressbar(
            init_iter = 0,
            final_iter = 2,
            line_size = 31,
            dur_queue_size = None,
        )
        pbar2 = CliProgressbar(
            init_iter = 0,
            final_iter = 2,
            line_size = 31*2+3,
            dur_queue_size = None,
            parent_bar = pbar1,
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar1.start()
        self.assertRegex(
            s.getvalue(),
            wiggle('  0% 0/2 00s ETA-??s@      s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar1.update(1)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 50% 1/2 01s ETA-01s@   1.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar2.start()
        self.assertRegex(
            s.getvalue(),
            wiggle(' 50% 1/2 02s ETA-02s@   2.0s/it >   0% 0/2 00s ETA-??s@      s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar2.update(1)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 50% 1/2 03s ETA-03s@   3.0s/it >  50% 1/2 01s ETA-01s@   1.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar2.update(2)
        self.assertRegex(
            s.getvalue(),
            wiggle(' 50% 1/2 04s ETA-04s@   4.0s/it > 100% 2/2 02s ETA-00s@   1.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar2.close()
        self.assertRegex(
            s.getvalue(),
            wiggle('                                                                 \r\
 50% 1/2 05s ETA-05s@   5.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            time.sleep(1.0)
            pbar1.update(2)
        self.assertRegex(
            s.getvalue(),
            wiggle('100% 2/2 06s ETA-00s@   3.0s/it\r')
        )

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            pbar1.close()
        self.assertRegex(
            s.getvalue(),
            wiggle('                               \r')
        )


#########################################
if __name__ == '__main__':
    unittest.main()
