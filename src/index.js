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
import { tfCam } from "./tfCamera";
tfjs.enableProdMode();

let score = 0;
let timeLabel = 0;
const video = document.getElementById("video");
const vidW = 640;
const vidH = 480;

const { ImageData } = canvas;
const TinyFaceDetectorOption = new faceapi.TinyFaceDetectorOptions({
  inputSize: 160,
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
  const canvas = faceapi.createCanvasFromMedia(video);
  const ctx = canvas.getContext("2d");
  document.body.append(canvas);
  const displaySize = { width: vidW, height: vidH };
  faceapi.matchDimensions(canvas, displaySize);

  tfCam.setCam(video);
  setInterval(async () => {
    if (timeLabel % 10 == 0) console.time("time fd" + timeLabel);

    const [detection, img] = await Promise.all([
      faceapi.detectSingleFace(video, TinyFaceDetectorOption),
      tfCam.capture([1, vidH, vidW, 3]),
    ]);

    if (timeLabel % 10 == 0) console.timeEnd("time fd" + timeLabel);
    if (timeLabel % 10 == 0) console.time("time lm" + timeLabel);

    const box = detection ? detection._box : undefined;
    const croppedFace = cropFace(box, img);
    const landmarks = landmarkModel.execute(croppedFace, "Identity_2");
    convertLandmark(landmarks, box).then((landmarkObj) => {
      score = analyze(detection, landmarkObj);
      drawAll(canvas, ctx, detection, landmarkObj, displaySize, score);
    });

    tfjs.dispose(landmarks);
    tfjs.dispose(croppedFace);

    if (timeLabel % 10 == 0) console.timeEnd("time lm" + timeLabel);
    timeLabel += 1;

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
