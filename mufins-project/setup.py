'''
Use "pip install -e ." in terminal to install this project.
'''
import os
import setuptools
import setuptools_scm

version_string = setuptools_scm.get_version(
    root=".",
    relative_to=__file__,
    local_scheme="node-and-timestamp",
)
with open(os.path.join('mufins', 'version.txt'), 'w', encoding='utf-8') as f:
    f.write(version_string)

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().strip().split('\n')

setuptools.setup(
    name='mufins',
    version=version_string,
    packages=[
        'mufins',
        'mufins.common',
        'mufins.resources',
    ],
    package_data={
        'mufins': [
            'version.txt',
        ]
    },
    install_requires=requirements
)
