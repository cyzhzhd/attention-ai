/* use tfjs-node */
const os = require("os");
var fs = require("fs");
const path = require("path");
const faceapi = require("face-api.js");
const tfjs = require("@tensorflow/tfjs-node");
const { performance } = require("perf_hooks");
const blazeface = require("@tensorflow-models/blazeface");
tfjs.enableProdMode();

const macDir = "../../Widerface/WIDER_val/images";
const lnxDir = "../../hddrive/WIDER_val/images";

const backend = "tensorflow";
const testName = "128-0.1-Blazeface";
const testPath = path.join(__dirname, os.type() == "Linux" ? lnxDir : macDir);
const outputPath = path.join(__dirname, "./result-" + testName);
const modelPath = path.join(__dirname, "../dist/models-faceapi");
let blazefaceModel = null;
let detected = 0;
let imgNum = 0;
let timeTaken = 0;

const TinyFaceDetectorOption = new faceapi.TinyFaceDetectorOptions({
  inputSize: 640,
  scoreThreshold: 0.1,
});

// SsdMobilenetv1 works with tfjs 1.7.x
const SsdMobilenetv1Option = new faceapi.SsdMobilenetv1Options({
  minConfidence: 0.1,
  maxResults: 987654321,
});

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

async function processFolder(testFolder, resultFolder) {
  console.log(`Working on ${testFolder}`);
  const testFiles = await getFiles(testFolder);
  for await (const file of testFiles) {
    const fileName = file.split(".")[0];
    const resultFile = path.join(resultFolder, fileName + ".txt");
    fs.writeFileSync(resultFile, fileName + "\n");
    const imgBuffer = fs.readFileSync(path.join(testFolder, file));
    const img = tfjs.node.decodeImage(new Uint8Array(imgBuffer));

    const t0 = performance.now();
    const detectResult = await blazefaceDetect(img);
    /*
    const detectResult = await faceapi.detectAllFaces(
      img,
      SsdMobilenetv1Option // set your faceapi model here
    );
    */
    const t1 = performance.now();

    // ignore first one, abnormal time
    timeTaken = timeTaken + (imgNum < 0 ? 0 : t1 - t0);
    imgNum = imgNum + 1;
    detected += detectResult.length;

    fs.writeFileSync(resultFile, `${detectResult.length}\n`, { flag: "a" });
    for await (const detection of detectResult) {
      const { _box, _score } = detection;
      fs.writeFileSync(
        resultFile,
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
}

async function blazefaceDetect(img) {
  const predictions = await blazefaceModel.estimateFaces(img, false);
  const result = predictions.map((obj) => {
    return {
      _score: obj.probability[0],
      _box: {
        _x: obj.topLeft[0],
        _y: obj.topLeft[1],
        _width: obj.bottomRight[0] - obj.topLeft[0],
        _height: obj.bottomRight[1] - obj.topLeft[1],
      },
    };
  });
  return result;
}

async function main() {
  await tfjs.setBackend(backend);
  console.log(`Start testing: ${testName}, ${tfjs.getBackend()}`);
  const testFolders = await getFiles(testPath);
  await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  blazefaceModel = await blazeface.load({ scoreThreshold: 0.1 });
  makeDirIfNotExists(outputPath);
  for await (const folder of testFolders) {
    const testFolder = path.join(testPath, folder);
    const resultFolder = path.join(outputPath, folder);
    makeDirIfNotExists(resultFolder);
    await processFolder(testFolder, resultFolder);
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
