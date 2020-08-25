"""
TODO: lazy(dynamic loading data), data augmentation
"""

from widerface_loader import load_widerface, generate_gt, dataloader
from model.models import Blazeface
from losses import MultiboxLoss
import tensorflow as tf
import configobj
import numpy as np
import os

CONFIG_FILE = 'train_config.ini'


def preprocess_config():
    config = configobj.ConfigObj(CONFIG_FILE)
    cfg = config['DEFAULT']
    return cfg


if __name__ == "__main__":
    cfg = preprocess_config()
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", len(physical_devices))

    # [num_picture, width(128), height(128), 3], [num_picture, num_gt, 4]
    images, labels = load_widerface(cfg['wider_gt'], cfg['wider_train'])

    # [num_box(896), 4]
    anchors = np.load(os.path.join(cfg['anchor_path'], "anchors.npy"))

    # [num_labels, num_boxes, 5(conf, tcx, tcy, tw, th)]
    ground_truths = generate_gt(
        labels, anchors, cfg.as_float('gt_iou'))

    data_loader = dataloader(images, ground_truths,
                             batch_size=cfg.as_int('batch_size'))

    model = Blazeface(input_dim=(
        cfg.as_int('input_w'), cfg.as_int('input_h'), 3)).build_model()
    loss = MultiboxLoss
    optim = tf.keras.optimizers.Adam(
        learning_rate=cfg.as_float('learning_rate'))
    model.compile(loss=loss, optimizer=optim)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            cfg['model_path'], cfg['model_name'] + '{epoch:03d}.hdf5'),
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=cfg.as_float('early_stop_patience'))
    rdlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                factor=cfg.as_float(
                                                    'rdlr_factor'),
                                                patience=cfg.as_float(
                                                    'rdlr_patience'),
                                                min_lr=cfg.as_float('rdlr_min'))

    res = model.fit(x=data_loader,
                    steps_per_epoch=cfg.as_int('steps_per_epoch'),
                    epochs=cfg.as_int('epochs'),
                    verbose=cfg.as_int('verbose'),
                    callbacks=[ckpt, early_stop, rdlr])
