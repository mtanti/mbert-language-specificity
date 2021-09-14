'''
Create a legend that displays the association between labels and colours.

A legend consists of a table of cells where each cell consists of a patch of colour and a text
label.
To organise the cells, the maximum width and height of the labels is determined first, referred to
here as t_w and t_h respectively.
This is how a cell is organised:

- Patch of size (t_h, t_h).
- Space of size (t_h, PATCH_TO_LABEL_SPACE).
- Label of size (t_h, t_w).

Columns of cells are separated by a space of width COL_TO_COL_SPACE and rows are separated by a
space of ROW_TO_ROW_SPACE.
The labels are positioned top-down left-right such that the first column must be completely full
before the second column is used and the column is filled from top to bottom.

The number of rows and columns is determined by the combination that best results in a square
figure shape.
'''

import math
from contextlib import contextmanager
from typing import Sequence, Iterator, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches
from mufins.common.chart.colours import COLOURS


#########################################
FONT_SIZE = 14
PATCH_TO_LABEL_SPACE = 10
ROW_TO_ROW_SPACE = 10
COL_TO_COL_SPACE = 20


#########################################
def _get_fig_size(
    num_cols: int,
    num_rows: int,
    text_width: int,
    text_height: int,
) -> Tuple[int, int]:
    '''
    Predict the size of the resulting figure from the properties of the legend.

    :param num_cols: The number of columns in the legend.
    :param num_rows: The number of rows in the legend.
    :param text_width: The number of pixels reserved for label width space.
    :param text_height: The number of pixels reserved for label height space.
    :return: A tuple consisting of the figure width and figure height in pixels.
    '''
    cell_width = text_height + PATCH_TO_LABEL_SPACE + text_width
    cell_height = text_height

    fig_width = num_cols*(cell_width + COL_TO_COL_SPACE) - COL_TO_COL_SPACE
    fig_height = num_rows*(cell_height + ROW_TO_ROW_SPACE) - ROW_TO_ROW_SPACE

    return (fig_width, fig_height)

#########################################
def _get_patch_pos(
    index: int,
    num_cols: int,
    num_rows: int,
    text_width: int,
    text_height: int,
) -> Tuple[int, int]:
    '''
    Predict the coordinate of a colour patch from the properties of the legend.

    :param index: The index of the label.
    :param num_cols: The number of columns in the legend.
    :param num_rows: The number of rows in the legend.
    :param text_width: The number of pixels reserved for label width space.
    :param text_height: The number of pixels reserved for label height space.
    :return: A tuple consisting of the bottom-left (x, y) coordinate in pixels.
    '''
    if num_cols > 1:
        col = index//num_rows
        row = index%num_rows
    else:
        col = 0
        row = index

    cell_width = text_height + PATCH_TO_LABEL_SPACE + text_width
    cell_height = text_height

    x = col*(cell_width + COL_TO_COL_SPACE)
    y = (num_rows - 1 - row)*(cell_height + ROW_TO_ROW_SPACE)

    return (x, y)


#########################################
def _get_text_pos(
    index: int,
    num_cols: int,
    num_rows: int,
    text_width: int,
    text_height: int,
) -> Tuple[int, int]:
    '''
    Predict the coordinate of a text label from the properties of the legend.

    :param index: The index of the label.
    :param num_cols: The number of columns in the legend.
    :param num_rows: The number of rows in the legend.
    :param text_width: The number of pixels reserved for label width space.
    :param text_height: The number of pixels reserved for label height space.
    :return: A tuple consisting of the bottom-left (x, y) coordinate in pixels.
    '''
    patch_width = text_height
    (x, y) = _get_patch_pos(index, num_cols, num_rows, text_width, text_height)
    x += patch_width + PATCH_TO_LABEL_SPACE

    return (x, y)


#########################################
@contextmanager
def get_legend(
    label_vocab: Sequence[str],
) -> Iterator[plt.Figure]:
    '''
    Get a matplotlib figure showing a legend of colours to label names.

    :param label_vocab: A list of unique label names.
        The order will determine their colour.
    :return: Matplotlib figure with just a legend.
    '''
    fig = None
    try:
        num_labels = len(label_vocab)

        (fig, axis) = plt.subplots(1, 1)
        renderer = fig.canvas.get_renderer()

        texts = []
        text_dims = []
        for label_text in label_vocab:
            text = axis.text(
                0, 0,
                label_text,
                fontsize=FONT_SIZE,
                horizontalalignment='left',
                verticalalignment='bottom'
            )
            texts.append(text)
            box = text.get_window_extent(renderer=renderer)
            text_dim = (math.ceil(box.width), math.ceil(box.height))
            text_dims.append(text_dim)

        text_width = max(w for (w, h) in text_dims)
        text_height = max(h for (w, h) in text_dims)

        patches = []
        for colour in COLOURS[:num_labels]:
            patch = axis.add_patch(
                matplotlib.patches.Rectangle(
                    xy=(0, 0),
                    width=text_height, height=text_height,
                    facecolor=colour,
                    edgecolor='black'
                )
            )
            patches.append(patch)

        best = None
        min_diff: Optional[float] = None
        for num_cols in range(1, num_labels + 1):
            num_rows = math.ceil(num_labels/num_cols)
            (fig_width, fig_height) = _get_fig_size(num_cols, num_rows, text_width, text_height)
            diff = abs(fig_width - fig_height)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                best = (num_cols, num_rows, fig_width, fig_height)
        assert best is not None
        (num_cols, num_rows, fig_width, fig_height) = best

        for i in range(num_labels):
            patch = patches[i]
            (x, y) = _get_patch_pos(i, num_cols, num_rows, text_width, text_height)
            patch.set_x(x)
            patch.set_y(y + 1) # See Note 1 below.

            text = texts[i]
            (x, y) = _get_text_pos(i, num_cols, num_rows, text_width, text_height)
            text.set_x(x)
            text.set_y(y + 1)

        fig.set_dpi(72)
        fig.set_figwidth(fig_width/72)
        fig.set_figheight(fig_height/72)
        fig.subplots_adjust(0, 0, 1, 1)
        axis.set_xlim(0, fig_width)
        axis.set_ylim(0, fig_height + 1) # See Note 1 below.
        axis.axis('off')

        # Note 1:
        # The +1 was necessary to avoid having a missing row of pixels at the bottom of the figure.

        yield fig
    finally:
        if fig is not None:
            plt.close(fig)
