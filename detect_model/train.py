from widerface_loader import load_widerface
from models import Blazeface
from loss import MultiboxLoss
import tensorflow as tf
import numpy as np
import os

# TODO: config to file
CONFIG = {
    "train_ratio": 0.75,
    "anchor_num": [2, 6],
    "cell_size": [16, 8],
    "output_path": "./"
}

# "~~/wider_face_train_bbx_gt.txt" format
WIDER_GT = "/home/cyrojyro/hddrive/wider_face_split/wider_face_train_bbx_gt.txt"
# "~~/WIDER_train/images" format
WIDER_TRAIN = "/home/cyrojyro/hddrive/WIDER_train/images"


def dataloader(images, labels, batch_size=64):
    data_keys = np.arange(len(images))
    while True:
        selected_keys = np.random.choice(
            data_keys, replace=False, size=batch_size)

        image_batch = []
        label_batch = []
        for key in selected_keys:
            image_batch.append(images[key])
            label_batch.append(labels[key])

        yield (np.array(image_batch, dtype=np.float32), label_batch)


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", len(physical_devices))

    # [num_picture, width(128), height(128), 3], [num_picture, num_gt, 4]
    images, labels = load_widerface(WIDER_GT, WIDER_TRAIN)

    """ train validation split
    num_images = images.shape[0]
    num_train = int(num_images * CONFIG['train_ratio'])
    print("train: ", num_train, " val: ", num_images - num_train)
    train_images, train_labels = images[:num_train], labels[:num_train]
    val_images, val_labels = images[num_train:], labels[num_train:]
    """

    # [num_box(896), 4]
    anchors = np.load(os.path.join(CONFIG['output_path'], "anchors.npy"))

    # DO TRAIN
    model = Blazeface(input_dim=(128, 128, 3))
    loss = MultiboxLoss()
    optim = tf.keras.optimizers.Adam()
    model.compile(loss=loss, optimizer=optim)

    data_loader = dataloader(images, labels)
    res = model.fit(x=data_loader,
                    steps_per_epoch=100,
                    epochs=10,
                    verbose=1)
