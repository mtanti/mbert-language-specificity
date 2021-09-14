#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of MUFINS Project.
'''
Extract the colour list from http://godsnotwheregodsnot.blogspot.com/ which was constructed
to make each next colour as perceptually different from the previous colours as possible.
'''

import os
import urllib.request
import re
import json
import argparse
from typing import Tuple
import mufins


#########################################
def hex_to_rgb(
    hex_colour: str,
) -> Tuple[int, int, int]:
    '''
    Convert a hex colour format to RGB format.

    Example: #00FF00 -> (0, 255, 0)

    :param hex_colour: The hexidecimal format colour with a leading '#'.
    :return: The RGB triple of integers.
    '''
    hex_colour = hex_colour[1:] # Remove '#'.

    hex_r = hex_colour[0:2]
    hex_g = hex_colour[2:4]
    hex_b = hex_colour[4:6]
    return (
        int(hex_r, 16),
        int(hex_g, 16),
        int(hex_b, 16),
    )


#########################################
def main(
) -> None:
    '''
    Scrape the colour list from a website and save them in a JSON file.
    '''
    # Scrape the colour list.
    url = 'http://godsnotwheregodsnot.blogspot.com/2013/11/kmeans-color-quantization-seeding.html'
    with urllib.request.urlopen(url) as f:
        html = f.read().decode('utf-8')
    colours_section = re.search(r'new String\[\]\{([^}]*)\}', html)
    assert colours_section is not None
    colour_strings = re.findall('#[0-9A-F]{6}', colours_section.group(1))
    colour_rgbs = [hex_to_rgb(colour_string) for colour_string in colour_strings]

    with open(os.path.join(
        mufins.path, 'resources', 'colours', 'colours.json'
    ), 'w', encoding='utf-8') as f:
        json.dump(colour_rgbs, f)


#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Extract the colour list from http://godsnotwheregodsnot.blogspot.com/ which was'
            ' constructed to make each next colour as perceptually different from the previous'
            ' colours as possible.'
        )
    )
    args = parser.parse_args()

    main()
