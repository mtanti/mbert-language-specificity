'''
Load the colour list in the colours.json resource.
'''

import json
import pkg_resources


COLOURS = [
    tuple(channel/255 for channel in rgb)
    for rgb in json.loads(
        pkg_resources.resource_string('mufins.resources.colours', 'colours.json').decode()
    )
]
