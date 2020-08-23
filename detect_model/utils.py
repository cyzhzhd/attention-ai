import matplotlib.pyplot as plt
import matplotlib.patches as patches


def drawplt(image, label):
    """
    image: numpy array
    label: [num_face, 4], cx, cy, w, h: pixel location
    """
    fig, ax = plt.subplots(1)
    ax = plt.imshow(image)
    for l in label:
        rect = patches.Rectangle((l[0], l[1]), l[2], l[3], linewidth=1,
                                 edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.show()
