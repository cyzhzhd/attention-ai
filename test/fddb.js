const os = require("os");
var fs = require("fs");
const path = require("path");
const faceapi = require("face-api.js");
const tfjs = require("@tensorflow/tfjs-node");
const { performance } = require("perf_hooks");

tfjs.enableProdMode();

const macDir = "../../Widerface/WIDER_train/images";
const lnxDir = "../../hddrive/WIDER_train/images";

const backend = "tensorflow";
const testName = "128-0.1-Ours-fddb";
const foldPath = path.join(
  __dirname,
  "../../hddrive/wider_face_split/FDDB-folds"
);
const testPath = path.join(__dirname, os.type() == "Linux" ? lnxDir : macDir);
const outputPath = path.join(__dirname, "./result-" + testName);
const modelPath = path.join(__dirname, "../dist/models-faceapi");
const oursPath = path.join(
  __dirname,
  "../dist/models-tfjs/detector_crafted/model.json"
);

let detected = 0;
let imgNum = 0;
let timeTaken = 0;

const TinyFaceDetectorOption = new faceapi.TinyFaceDetectorOptions({
  inputSize: 128,
  scoreThreshold: 0.1,
});

class DetectorModel {
  constructor(imgW, imgH) {
    this.model = null;
    this.imgW = imgW;
    this.imgH = imgH;
    this.anchors = generate_anchors(0.2, 0.9, [2, 6], [16, 8]);
  }

  async loadFromUri(url) {
    this.model = await tfjs.loadGraphModel(url);
  }

  async predict(image, imgW, imgH) {
    const normalized = normalizeImage(image);
    const preds = this.model.predict(normalized);
    const pred = tfjs.squeeze(preds, 0);
    const [confs, bboxes] = prediction_to_bbox(pred, this.anchors, imgW, imgH);
    const args = tfjs.image.nonMaxSuppression(bboxes, confs, 100, 0.5, 0.1);

    const toCopy = [args, bboxes, confs];
    const [selectedArgs, boxes, cofs] = await Promise.all(
      toCopy.map(async (item) => {
        const arr = await item.array();
        return arr;
      })
    );

    const results = [];
    for (const arg of selectedArgs) {
      boxes[arg].push(cofs[arg]);
      results.push(boxes[arg]);
    }
    tfjs.dispose([args, normalized, preds, pred, confs, bboxes]);
    return results;
  }
}

const ours = new DetectorModel(640, 480);

function normalizeImage(image) {
  const normalized = tfjs.tidy(() => {
    return tfjs.sub(tfjs.div(image, 127.5), 1.0);
  });
  return normalized;
}

function generate_anchors(sMin, sMax, anchorNum, cellSize) {
  let totalNum = anchorNum.reduce((acc, val) => acc + val, 0);
  let anchors = new Array();
  let cellAcc = 0;

  for (let iter = 0; iter < anchorNum.length; iter++) {
    let cells = cellSize[iter];
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        for (let order = 0; order < anchorNum[iter]; order++) {
          let scale =
            sMin + ((sMax - sMin) / (totalNum - 1)) * (order + cellAcc);
          anchors.push([(x + 0.5) / cells, (y + 0.5) / cells, scale, scale]);
        }
      }
    }
    cellAcc += anchorNum[iter];
  }
  return tfjs.tensor(anchors);
}

function prediction_to_bbox(prediction, anchors, imgW, imgH) {
  const result = tfjs.tidy(() => {
    const anchorScales = tfjs.slice(anchors, [0, 2], [anchors.shape[0], 2]);
    const anchorCoords = tfjs.slice(anchors, [0, 0], [anchors.shape[0], 2]);
    const predScales = tfjs.slice(prediction, [0, 3], [prediction.shape[0], 2]);
    const predCoords = tfjs.slice(prediction, [0, 1], [prediction.shape[0], 2]);
    const predConfs = tfjs
      .slice(prediction, [0, 0], [prediction.shape[0], 1])
      .squeeze(-1);
    const imgSize = tfjs.tensor([imgW, imgH]);
    const center = tfjs
      .add(tfjs.mul(predCoords, anchorScales), anchorCoords)
      .mul(imgSize);
    const widthHeight = tfjs
      .mul(anchorScales, tfjs.exp(predScales))
      .mul(imgSize);

    const topLeft = tfjs.sub(center, tfjs.div(widthHeight, 2));
    const downRight = tfjs.add(center, tfjs.div(widthHeight, 2));
    const coords = tfjs.concat([topLeft, downRight], -1);
    return [predConfs, coords];
  });
  return result;
}

main();

function getFiles(path) {
  return new Promise((resolve) => {
    fs.readdir(path, function (err, files) {
      if (err) {
        return console.log("Unable to scan directory: " + err);
      }
      resolve(files);
    });
  });
}

function makeDirIfNotExists(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }
}

async function processImage(imgPath, imgName, savePath) {
  const imgBuffer = fs.readFileSync(imgPath);
  let img = tfjs.node.decodeImage(new Uint8Array(imgBuffer));
  if (img.shape[2] == 1) {
    const new_img = tfjs.concat([img, img, img], -1);
    tfjs.dispose(img);
    img = new_img;
  }
  const t0 = performance.now();
  const detectResult = await detectOurs(img);
  /*
  const detectResult = await faceapi.detectAllFaces(
    img,
    TinyFaceDetectorOption // set your faceapi model here
  );
  */
  const t1 = performance.now();
  // ignore first one, abnormal time
  timeTaken = timeTaken + (imgNum < 0 ? 0 : t1 - t0);
  imgNum = imgNum + 1;
  detected += detectResult.length;
  savePath = savePath + "/" + imgName.split("/").join("_") + ".txt";
  fs.writeFileSync(savePath, `${imgName.split("/").join("_")}\n`, {
    flag: "a",
  });
  fs.writeFileSync(savePath, `${detectResult.length}\n`, { flag: "a" });
  for await (const detection of detectResult) {
    const { _box, _score } = detection;
    fs.writeFileSync(
      savePath,
      `${Math.round(_box._x)} ${Math.round(_box._y)} ${Math.round(
        _box._width
      )} ${Math.round(_box._height)} ${_score.toFixed(3)}\n`,
      {
        flag: "a",
      }
    );
  }

  tfjs.dispose(img);
}

async function detectOurs(img) {
  const detectImg = tfjs.image.resizeBilinear(img, [128, 128]);
  const squeezed = tfjs.expandDims(detectImg, 0);
  const predictions = await ours.predict(squeezed, img.shape[1], img.shape[0]);
  const result = predictions.map((obj) => {
    return {
      _score: obj[4],
      _box: {
        _x: obj[0],
        _y: obj[1],
        _width: obj[2] - obj[0],
        _height: obj[3] - obj[1],
      },
    };
  });
  tfjs.dispose([detectImg, squeezed]);
  return result;
}

async function main() {
  await tfjs.setBackend(backend);
  console.log(`Start testing: ${testName}, ${tfjs.getBackend()}`);
  await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
  await ours.loadFromUri("file://" + oursPath);
  makeDirIfNotExists(outputPath);

  const folds = await getFiles(foldPath);
  folds.sort();
  for await (const txtFile of folds) {
    console.log("Working on", txtFile);
    const curPath = path.join(
      outputPath,
      parseInt(txtFile.split("-")[2].split(".")[0]).toString()
    );
    makeDirIfNotExists(curPath);
    const txtFilePath = path.join(foldPath, txtFile);
    const data = fs.readFileSync(txtFilePath, "UTF-8");
    const lines = data.split(/\r?\n/);
    for await (const line of lines) {
      if (line == "") continue;
      const imgPath = path.join(testPath, line) + ".jpg";
      await processImage(imgPath, line, curPath);
    }
    console.log(`Current time: ${timeTaken}ms, Processed Iamges: ${imgNum}`);
  }

  console.log(`${detected} face detected.`);
  console.log(`${tfjs.getBackend()} backend used.`);
  console.log(`Avg time taken: ${timeTaken / (imgNum - 1)}ms`);
  fs.writeFileSync(
    "./timeResult.txt",
    `${testName} ${tfjs.getBackend()} ${
      timeTaken / imgNum
    }ms ${detected} face detected.\n`,
    { flag: "a" }
  );
}
