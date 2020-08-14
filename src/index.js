/*
TODO: 
      preprocess face for better results,
      adaptive method to analyze face
      (use facebox, overall median, face calibration phase etc...),
      analyze sight,
      development/production setting
*/
import * as canvas from "canvas";
import * as faceapi from "face-api.js";
import * as tfjs from "@tensorflow/tfjs";
import { status, analyze } from "./analysis.js";
import { landmarkModel, convertLandmark } from "./landmark.js";
tfjs.enableProdMode();

let score = 0;
let frames = 0;
let task = null;
const upload = document.getElementById("myFileUpload");
const uploadedImg = document.getElementById("myImg");
const stop = document.getElementById("stopButton");
const video = document.getElementById("video");
const vidW = 640;
const vidH = 480;

stop.onclick = onStop;
upload.onchange = uploadTest;

const { ImageData } = canvas;
const TinyFaceDetectorOption = new faceapi.TinyFaceDetectorOptions({
  inputSize: 640,
  scoreThreshold: 0.3,
});

faceapi.env.monkeyPatch({
  Canvas: HTMLCanvasElement,
  Image: HTMLImageElement,
  ImageData: ImageData,
  Video: HTMLVideoElement,
  createCanvasElement: () => document.createElement("canvas"),
  createImageElement: () => document.createElement("img"),
});

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("./models-faceapi"),
  landmarkModel.loadFromUri("../dist/models-tfjs/keypoints_tfjs/model.json"),
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: { width: vidW, height: vidH } },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.addEventListener("play", async () => {
  console.log(`Backend: ${tfjs.getBackend()}`);
  const canvas = faceapi.createCanvasFromMedia(video);
  const ctx = canvas.getContext("2d");
  document.body.append(canvas);
  const displaySize = { width: vidW, height: vidH };
  faceapi.matchDimensions(canvas, displaySize);

  task = setInterval(async () => {
    const timefd1 = performance.now();
    const detection = await faceapi.detectSingleFace(
      video,
      TinyFaceDetectorOption
    );
    const timefd2 = performance.now();

    const timelm1 = performance.now();
    const box = detection ? detection._box : undefined;
    const pixel = tfjs.browser.fromPixels(video);
    const img = pixel.reshape([-1, vidH, vidW, 3]);
    const croppedFace = cropFace(box, img);
    const landmarks = landmarkModel.execute(croppedFace, "Identity_2");
    convertLandmark(landmarks, box).then((landmarkObj) => {
      score = analyze(detection, landmarkObj);
      drawAll(canvas, ctx, detection, landmarkObj, displaySize, score);
    });
    const timelm2 = performance.now();

    frames = frames + 1;
    if (frames % 10 === 0)
      console.log(
        `${frames}: fd ${(timefd2 - timefd1).toFixed(3)}ms lm ${(
          timelm2 - timelm1
        ).toFixed(3)}ms`
      );

    tfjs.dispose(landmarks);
    tfjs.dispose(croppedFace);
    tfjs.dispose(pixel);
    tfjs.dispose(img);
  }, 50);
});

function cropFace(box, img) {
  if (box === undefined) return undefined;
  return tfjs.image.cropAndResize(
    img,
    [
      [
        box.y / vidH,
        box.x / vidW,
        (box.y + box.height) / vidH,
        (box.x + box.width) / vidW,
      ],
    ],
    [0],
    [160, 160]
  );
}

function drawAll(canvas, ctx, detection, landmarkObj, displaySize, score) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (detection !== undefined) {
    const resizedDetections = faceapi.resizeResults(detection, displaySize);
    faceapi.draw.drawDetections(canvas, resizedDetections);
  }
  ctx.fillStyle = "#FF00FF";
  if (landmarkObj !== undefined) {
    for (let i = 0; i < 68; ++i) {
      ctx.fillRect(landmarkObj[i]["_x"], landmarkObj[i]["_y"], 3, 3);
    }
  }
  drawInfo(ctx, score);
}

function drawInfo(ctx, score) {
  ctx.fillStyle = "#FF00FF";
  ctx.font = "30px Arial";
  ctx.fillText("score: " + score, 30, 50);
  ctx.font = "18px Arial";
  var lines = JSON.stringify(status, null, 2).split("\n");
  for (var j = 0; j < lines.length; j++)
    ctx.fillText(lines[j], 10, 310 + j * 20);
  ctx.font = "12px Arial";
  ctx.fillText(JSON.stringify(tfjs.memory()), 20, 470);
}

function onStop() {
  video.pause();
  clearInterval(task);
}

async function uploadTest() {
  const imgFile = document.getElementById("myFileUpload").files[0];
  const img = await faceapi.bufferToImage(imgFile);
  document.getElementById("myImg").src = img.src;

  const t1 = performance.now();
  await faceapi.detectAllFaces(uploadedImg, TinyFaceDetectorOption);
  const t2 = performance.now();

  console.log(`took ${(t2 - t1).toFixed(3)} ms to process`);
}
