import matplotlib.pyplot as plt
import matplotlib.patches as patches


def drawplt(image, label, target_w, target_h):
    """
    image: numpy array
    label: [num_face, 4], cx, cy, w, h: ratio of pixel location
    """
    fig, ax = plt.subplots(1)
    ax = plt.imshow(image)
    for l in label:
        rect = patches.Rectangle((l[0] * target_w, l[1] * target_h),
                                 l[2] * target_w, l[3] * target_h, linewidth=1,
                                 edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.show()


# TODO np execution
def calc_iou_batch(box, batch):
    batch_result = []
    for b in batch:
        batch_result.append(calc_iou(box, b))
    return batch_result


def calc_iou(box_1, box_2):
    box_1_xmax = box_1[0] + (box_1[2] / 2)
    box_1_xmin = box_1[0] - (box_1[2] / 2)
    box_1_ymax = box_1[1] + (box_1[3] / 2)
    box_1_ymin = box_1[1] - (box_1[3] / 2)
    box_1_space = (box_1_ymax - box_1_ymin) * (box_1_xmax - box_1_xmin)

    box_2_xmax = box_2[0] + (box_2[2] / 2)
    box_2_xmin = box_2[0] - (box_2[2] / 2)
    box_2_ymax = box_2[1] + (box_2[3] / 2)
    box_2_ymin = box_2[1] - (box_2[3] / 2)
    box_2_space = (box_2_ymax - box_2_ymin) * (box_2_xmax - box_2_xmin)

    i_xmin = max(box_1_xmin, box_2_xmin)
    i_xmax = min(box_1_xmax, box_2_xmax)
    i_ymin = max(box_1_ymin, box_2_ymin)
    i_ymax = min(box_1_ymax, box_2_ymax)

    i_w = max(i_xmax - i_xmin, 0)
    i_h = max(i_ymax - i_ymin, 0)
    intersection = i_w * i_h

    return intersection / (box_1_space + box_2_space - intersection)
