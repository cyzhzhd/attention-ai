"""
TODO: __init__ module job, camera test code
"""

from utils.widerface_loader import *
from model.models import Blazeface
from utils.losses import MultiboxLoss
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

    gts = cfg['wider_gts']
    gts = [gts] if isinstance(gts, str) else [gt for gt in gts]

    # [num_picture, width(128), height(128), 3], [num_picture, num_gt, 4]
    images, labels = load_widerface(gts, cfg['wider_train'], cfg.as_int('input_w'),
                                    cfg.as_int('input_h'), max_size=cfg.as_int('max_dset'))

    # [num_box(896), 4]
    anchors = np.load(os.path.join(cfg['anchor_path'], "anchors.npy"))

    # split validation set
    val_num = int(len(images) * cfg.as_float('validation_ratio'))
    images_val, labels_val = images[:val_num], labels[:val_num]
    images, labels = images[val_num:], labels[val_num:]

    data_loader = dataloader_aug(images, labels, anchors,
                                 batch_size=cfg.as_int('batch_size'))

    gt_val = generate_gt(labels_val, anchors, verbose=True)
    data_loader_val = dataloader_gt(images_val, gt_val,
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
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=cfg.as_float('early_stop_patience'))
    rdlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=cfg.as_float(
                                                    'rdlr_factor'),
                                                patience=cfg.as_float(
                                                    'rdlr_patience'),
                                                min_lr=cfg.as_float('rdlr_min'))

    res = model.fit(x=data_loader,
                    validation_data=data_loader_val,
                    validation_steps=cfg.as_int('validation_steps'),
                    epochs=cfg.as_int('epochs'),
                    steps_per_epoch=cfg.as_int('steps_per_epoch'),
                    verbose=cfg.as_int('verbose'),
                    callbacks=[ckpt, early_stop, rdlr])
