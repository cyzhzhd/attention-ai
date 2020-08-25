import argparse
import numpy as np
import os

CONFIG = {
    "s_min": 0.2,
    "s_max": 0.9,
    "anchor_num": [2, 6],
    "cell_size": [16, 8],
    "output_path": "./"
}


def calc_anchors(s_min, s_max, anchor_num, cell_size):
    """
    [num_box, 4(cx, cy, w, h)] anchor(prior) creation.\n
    w, h ratio fixed to 1 (following Blazeface paper)
    """
    total_anchor = sum(anchor_num)

    anchors = np.empty([0, 4], dtype=np.float32)
    cell_cumulated = 0
    for iteration, anchor in enumerate(anchor_num):
        cells = cell_size[iteration]
        for y in range(cells):
            for x in range(cells):
                for order in range(anchor):
                    scale = s_min + (s_max - s_min) / \
                        (total_anchor - 1) * (order + cell_cumulated)
                    anchors = np.vstack([anchors, np.array(
                        [(x + 0.5) / cells, (y + 0.5) / cells, scale, scale])])
        cell_cumulated += anchor_num[iteration]
    return anchors


if __name__ == "__main__":
    s_min = CONFIG['s_min']
    s_max = CONFIG['s_max']
    anchor_num = CONFIG['anchor_num']
    cell_size = CONFIG['cell_size']

    anchors = calc_anchors(s_min, s_max, anchor_num, cell_size)

    print("last 10 results:", anchors[-10:], sep='\n')
    np.save(os.path.join(CONFIG['output_path'], "anchors.npy"), anchors)
