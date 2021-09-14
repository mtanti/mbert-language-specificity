'''
Create a scatter plot.
'''

from contextlib import contextmanager
from typing import Sequence, Iterator
import matplotlib.pyplot as plt
from mufins.common.random.random_number_generator import RandomNumberGenerator
from mufins.common.chart.colours import COLOURS


#########################################
@contextmanager
def get_scatter(
    label_vocab: Sequence[str],
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    shuffle: bool = True,
) -> Iterator[plt.Figure]:
    '''
    Get a matplotlib figure showing a scatter plot.

    :param label_vocab: A list of unique label names.
        The order will determine their colour.
    :param x: The x-coordinates of the scatter points.
    :param y: The y-coordinates of the scatter points.
    :param labels: The label names of each point.
    :param shuffle: Whether to shuffle the points prior to plotting them to avoid overlapping
        points being determined by order.
    :return: Matplotlib figure with just a legend.
    '''
    if not len(x) == len(y) == len(labels):
        raise ValueError('Note the same number of x-values, y-values, and labels.')

    label2index = {label: i for (i, label) in enumerate(label_vocab)}
    point_colours = [COLOURS[label2index[label]] for label in labels]

    indexes = list(range(len(labels)))
    if shuffle:
        rng = RandomNumberGenerator()
        rng.shuffle(indexes)

    fig = None
    try:
        (fig, axis) = plt.subplots(1, 1)
        axis.scatter(
            [x[i] for i in indexes],
            [y[i] for i in indexes],
            c=[point_colours[i] for i in indexes],
            edgecolors='black', linewidths=0.5,
        )
        plt.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False,
        )
        axis.grid()
        fig.tight_layout(pad=0.1)

        yield fig
    finally:
        if fig is not None:
            plt.close(fig)
