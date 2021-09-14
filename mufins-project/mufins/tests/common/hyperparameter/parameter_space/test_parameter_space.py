'''
Unit test for Parameter Space class in hyperparameter module.
'''

import os
import unittest
import tempfile
from mufins.common.hyperparameter.parameter_space import AttributesType, ParameterSpace


#########################################
class TestParameterSpace(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the Parameter Space class.
        '''

        class Test(ParameterSpace[str]):
            # pylint: disable=missing-class-docstring
            # pylint: disable=missing-function-docstring
            # pylint: disable=no-self-use
            # pylint: disable=too-few-public-methods
            def parameter_decoder_full(
                self,
                full_attributes: AttributesType,
            ) -> str:
                return str(full_attributes['x'])

        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, 'file.txt'), 'w', encoding='utf-8') as f:
                print('''\
a\t{"x": 1}
b\t{"x": 2}
''', file=f)

            obj = Test(
                default_attributes={},
                attributes_list_or_path=os.path.join(path, 'file.txt'),
            )

            self.assertEqual(
                list(obj),
                [
                    ('a', '1'),
                    ('b', '2'),
                ],
            )


#########################################
if __name__ == '__main__':
    unittest.main()
