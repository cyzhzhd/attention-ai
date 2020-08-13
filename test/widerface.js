var fs = require("fs");
const path = require("path");
const faceapi = require("face-api.js");
const tfjs = require("@tensorflow/tfjs-node");
const { TinyFaceDetectorOptions } = require("face-api.js");

const testPath = path.join(__dirname, "../../hddrive/WIDER_val/images");
const outputPath = path.join(__dirname, "./result-512-0.1-Tiny");
const modelPath = path.join(__dirname, "../dist/models-faceapi");
let detected = 0;

const TinyFaceDetectorOption = new faceapi.TinyFaceDetectorOptions({
  inputSize: 512,
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
  console.log(`working on ${testFolder}`);
  const testFiles = await getFiles(testFolder);
  for await (const file of testFiles) {
    const fileName = file.split(".")[0];
    const resultFile = path.join(resultFolder, fileName + ".txt");
    fs.writeFileSync(resultFile, fileName + "\n");
    const imgBuffer = fs.readFileSync(path.join(testFolder, file));
    const img = tfjs.node.decodeImage(new Uint8Array(imgBuffer));

    const detectResult = await faceapi.detectAllFaces(
      img,
      TinyFaceDetectorOption
    );

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

async function main() {
  const testFolders = await getFiles(testPath);
  await faceapi.nets.tinyFaceDetector.loadFromDisk(modelPath);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  makeDirIfNotExists(outputPath);
  for await (const folder of testFolders) {
    const testFolder = path.join(testPath, folder);
    const resultFolder = path.join(outputPath, folder);
    makeDirIfNotExists(resultFolder);
    await processFolder(testFolder, resultFolder);
  }
  console.log(`${detected} face detected.`);
}
