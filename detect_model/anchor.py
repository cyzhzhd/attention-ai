import numpy as np
import os

# TODO: config to file
CONFIG = {
    "anchor_num": [2, 6],
    "cell_size": [16, 8],
    "output_path": "./"
}

# [num_box(896), 4(cx, cy, w, h)] anchor creation
# w, h fixed to 1 (1:1 ratio + no size difference)
if __name__ == "__main__":
    anchors = np.empty([0, 4], dtype=np.float32)
    anchor_num = CONFIG['anchor_num']
    cell_size = CONFIG['cell_size']

    for iteration, anchor in enumerate(anchor_num):
        cells = cell_size[iteration]
        for y in range(cells):
            for x in range(cells):
                for _ in range(anchor):
                    anchors = np.vstack([anchors, np.array(
                        [(1 * x + 0.5) / cells, (1 * y + 0.5) / cells, 1, 1])])

    np.save(os.path.join(CONFIG['output_path'], "anchors.npy"), anchors)
