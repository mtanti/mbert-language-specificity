'''
Project-level __init__.py.
'''

import os
import pkg_resources

__version__ = pkg_resources.resource_string(
    'mufins',
    'version.txt',
).decode()

path = os.path.dirname(os.path.abspath(__file__))
