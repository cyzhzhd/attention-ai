import * as tfjs from "@tensorflow/tfjs";

class LandmarkModel {
  constructor() {
    this.model = null;
    this.loadModel = this.loadFromUri.bind(this);
    this.execute = this.execute.bind(this);
  }

  async loadFromUri(url) {
    await tfjs.loadGraphModel(url).then((loadedModel) => {
      this.model = loadedModel;
    });
    return new Promise((resolve) => {
      resolve();
    });
  }

  execute(croppedFace, outNode) {
    if (croppedFace === undefined) return undefined;
    return this.model.execute(croppedFace, outNode);
  }
}

export async function convertLandmark(landmarks, box) {
  if (landmarks === undefined) return undefined;

  const _landmarks = await landmarks.array();
  let landmarkObj = new Array();
  for (let i = 0; i < 136; ++i) {
    if (!(i % 2)) landmarkObj[Math.floor(i / 2)] = new Object();

    let o =
      i % 2
        ? { key: "_y", offset: box.y, mult: box.height }
        : { key: "_x", offset: box.x, mult: box.width };
    landmarkObj[Math.floor(i / 2)][o.key] =
      o.offset + _landmarks[0][i] * o.mult;
  }
  return landmarkObj;
}

export const landmarkModel = new LandmarkModel();
