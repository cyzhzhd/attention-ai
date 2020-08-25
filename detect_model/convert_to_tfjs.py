import tensorflowjs as tfjs
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(
    description="Convert model to tfjs compatible")
parser.add_argument('--model', type=str, required=True,
                    help='Model dircetory')
parser.add_argument('--output', type=str, required=True,
                    help='Output directory')

if __name__ == "__main__":
    args = parser.parse_args()
    model = tf.keras.models.load_model(
        args.model, compile=False)
    tfjs.converters.save_keras_model(model, args.output)
