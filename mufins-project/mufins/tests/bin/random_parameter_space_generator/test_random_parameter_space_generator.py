'''
Unit test for random_parameter_space_generator script.
'''

import os
import unittest
import tempfile
import subprocess
import json
import mufins


#########################################
class TestRandomParameterSpaceGenerator(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_(
        self,
    ) -> None:
        '''
        Test the random_parameter_space_generator script.
        '''
        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, 'spec.json'), 'w', encoding='utf-8') as f:
                print('''\
{
    "a": {
        "dtype": "str",
        "values": ["a", "b"]
    },
    "b": {
        "dtype": "str",
        "values": ["c", "d"]
    },
    "c": {
        "dtype": "int",
        "values": [1, 2]
    },
    "d": {
        "dtype": "int",
        "values": [3, 4]
    },
    "e": {
        "dtype": "float",
        "values": [1.1, 2.2]
    },
    "f": {
        "dtype": "float",
        "values": [3.1, 4.2]
    },
    "g": {
        "dtype": "bool"
    }
}\
''', file=f)

            subprocess.run([
                'python',
                os.path.join(
                    mufins.path, '..', 'bin', 'random_parameter_space_generator.py'
                ),
                '--spec_file_path', os.path.join(path, 'spec.json'),
                '--output_file_path', os.path.join(path, 'output.txt'),
                '--amount', "2",
                '--seed', '0',
            ], check=True)

            with open(os.path.join(path, 'output.txt'), 'r', encoding='utf-8') as f:
                data = [
                    json.loads(line.split('\t')[1])
                    for line in f.read().strip().split('\n')
                ]
                self.assertNotEqual(data[0], data[1])
                for attrs in data:
                    self.assertIn(attrs['a'], ['a', 'b'])
                    self.assertIn(attrs['b'], ['c', 'd'])
                    self.assertIn(attrs['c'], [1, 2])
                    self.assertIn(attrs['d'], [3, 4])
                    self.assertIn(attrs['e'], [1.1, 2.2])
                    self.assertIn(attrs['f'], [3.1, 4.2])
                    self.assertIn(attrs['g'], [False, True])


#########################################
if __name__ == '__main__':
    unittest.main()
