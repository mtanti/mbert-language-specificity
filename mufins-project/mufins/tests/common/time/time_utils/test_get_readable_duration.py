'''
Unit test for get_readable_duration function in time utilities module.
'''

import unittest
from mufins.common.time.time_utils import get_readable_duration


#########################################
class TestGetReadableDuration(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the function.
        '''
        for (duration, readable) in [
            (0.0, '00s'),
            (0.4, '00s'),
            (0.9, '01s'),
            (1.0, '01s'),
            (10.0, '10s'),
            (60.0, '01m:00s'),
            (60.0+2.0, '01m:02s'),
            (2*60.0+2.0, '02m:02s'),
            (2*60*60+2*60.0+2.0, '02h:02m:02s'),
            (2*24*60*60+2*60*60+2*60.0+2.0, '2d:02h:02m:02s'),
            (10*24*60*60+2*60*60+2*60.0+2.0, '10d:02h:02m:02s'),
            (10*24*60*60+23*60*60+59*60.0+59.0, '10d:23h:59m:59s'),
            (10*24*60*60+23*60*60+59*60.0+59.9, '11d:00h:00m:00s'),
        ]:
            self.assertEqual(
                get_readable_duration(duration),
                readable,
                '{} | {}'.format(duration, readable),
            )


#########################################
if __name__ == '__main__':
    unittest.main()
