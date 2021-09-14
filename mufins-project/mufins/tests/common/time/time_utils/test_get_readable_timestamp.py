'''
Unit test for get_readable_timestamp function in time utilities module.
'''

import unittest
from mufins.common.time.time_utils import get_readable_timestamp


#########################################
class TestGetReadableTimestamp(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_given_timestamp(
        self,
    ) -> None:
        '''
        Test the function when an explicit timestamp is given.
        '''
        for (timestamp, readable) in [
            (0.0, '1970/01/01 00:00:00 GMT'),
            (0.99, '1970/01/01 00:00:00 GMT'),
            (1601137743.0, '2020/09/26 16:29:03 GMT'),
        ]:
            self.assertEqual(
                get_readable_timestamp(timestamp),
                readable,
                '{} | {}'.format(timestamp, readable),
            )

    #########################################
    def test_now(
        self,
    ) -> None:
        '''
        Test the function when the current timestamp is to be used.
        '''
        readable = get_readable_timestamp()
        self.assertRegex(readable, r'^[0-9]{4}\/[0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT$')
        readable2 = get_readable_timestamp()
        self.assertGreaterEqual(readable2, readable)


#########################################
if __name__ == '__main__':
    unittest.main()
