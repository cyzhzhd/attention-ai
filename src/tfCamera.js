import * as tfjs from "@tensorflow/tfjs";

class TFCamera {
  constructor() {
    this.cam = null;
    this.setCam = this.setCam.bind(this);
    this.capture = this.capture.bind(this);
  }

  async setCam(videoElement) {
    this.cam = await tfjs.data.webcam(videoElement);
  }

  capture(shape) {
    return new Promise((resolve, reject) => {
      try {
        this.cam.capture().then((img) => {
          const _img = tfjs.reshape(img, shape);
          tfjs.dispose(img);
          resolve(_img);
        });
      } catch (e) {
        reject(e);
      }
    });
  }
}

export const tfCam = new TFCamera();
