from widerface_loader import process_image, read_image
from utils import prediction_to_bbox, drawplt
from losses import MultiboxLoss
import tensorflow as tf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
    description="Test model using arbitrary image")
parser.add_argument('--width', type=int, default=128,
                    help='Target image width')
parser.add_argument('--height', type=int, default=128,
                    help='Target image height')
parser.add_argument('--anchor', type=str, default='./',
                    help='Path of generated anchors')
parser.add_argument('--threshold', type=float, default=0.7,
                    help='Threshold of detection')
parser.add_argument('--model', type=str, required=True,
                    help='Model dircetory')
parser.add_argument('--image', type=str, required=True,
                    help='Image dircetory')

if __name__ == "__main__":
    args = parser.parse_args()
    anchors = np.load(os.path.join(args.anchor, "anchors.npy"))

    model = tf.keras.models.load_model(
        args.model, custom_objects={'MultiboxLoss': MultiboxLoss})

    image = read_image(os.path.join(args.image), args.width, args.height)
    image_normalized = process_image(
        os.path.join(args.image), args.width, args.height)
    image_normalized = tf.convert_to_tensor(image_normalized)
    image_normalized = tf.expand_dims(image_normalized, 0)

    prediction = model.predict(image_normalized)
    bbox = prediction_to_bbox(prediction, anchors)[0]
    bbox = bbox[bbox[..., 0] > args.threshold]
    drawplt(image, bbox[..., 1:5], args.width, args.height)
