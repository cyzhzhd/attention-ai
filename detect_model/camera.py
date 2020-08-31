from utils.utils import prediction_to_bbox, tie_resolution
import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

parser = argparse.ArgumentParser(
    description="Test model using arbitrary image")
parser.add_argument('--width', type=int, default=128,
                    help='Target image width')
parser.add_argument('--height', type=int, default=128,
                    help='Target image height')
parser.add_argument('--anchor', type=str, default='./',
                    help='Path of generated anchors')
parser.add_argument('--threshold', type=float, default=0.3,
                    help='Threshold of detection')
parser.add_argument('--tie_threshold', type=float, default=0.2,
                    help='Tie threshold of predicted boxes')
parser.add_argument('--cpu', action="store_true",
                    help="Use CPU for inference")
parser.add_argument('--model', type=str, required=True,
                    help='Model directory')

if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
    args = parser.parse_args()
    anchors = np.load(os.path.join(args.anchor, "anchors.npy"))

    model = tf.keras.models.load_model(args.model, compile=False)

    physical_devices = tf.config.list_physical_devices('GPU')
    num_gpu = len(physical_devices)
    print("Available GPUs:", num_gpu)

    device = '/CPU:0' if (args.cpu or num_gpu == 0) else '/GPU:0'
    print('using ' + device)

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_resized = cv2.resize(frame, (args.width, args.height))
        frame_resized = frame_resized / 127.5 - 1.0
        prediction = model(np.expand_dims(frame_resized, 0))
        prediction = np.array(prediction)
        bbox = prediction_to_bbox(prediction, anchors)[0]
        bbox = bbox[bbox[..., 0] > args.threshold]
        resolved_boxes = tie_resolution(bbox, args.threshold, args.tie_threshold)
        
        if ret==True:
            for box in resolved_boxes:
                top_left = (int(frame.shape[1] * (box[0] - box[2] / 2)), int(frame.shape[0] * (box[1] - box[3] / 2)))
                botton_right = (int(frame.shape[1] * (box[0] + box[2] / 2)), int(frame.shape[0] * (box[1] + box[3] / 2)))
                blue = (255, 0, 0)
                frame = cv2.rectangle(frame, top_left, botton_right, blue, 2) 
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
