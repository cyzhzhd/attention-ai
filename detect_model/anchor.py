import numpy as np
import os

# TODO: config to file
CONFIG = {
    "s_min": 0.1,
    "s_max": 0.8,
    "anchor_num": [2, 6],
    "total_anchor": 8,
    "cell_size": [16, 8],
    "output_path": "./"
}

# [num_box(896), 4(cx, cy, w, h)] anchor creation, save to .npy file
# w, h ratio fixed to 1 (following blazeface paper)
if __name__ == "__main__":
    s_min = CONFIG['s_min']
    s_max = CONFIG['s_max']
    total_anchor = CONFIG['total_anchor']
    anchor_num = CONFIG['anchor_num']
    cell_size = CONFIG['cell_size']

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
                        [(1 * x + 0.5) / cells, (1 * y + 0.5) / cells, scale, scale])])
        cell_cumulated += anchor_num[iteration]

    print("last 10 results:")
    print(anchors[-10:])
    np.save(os.path.join(CONFIG['output_path'], "anchors.npy"), anchors)
