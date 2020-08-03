/*
TODO: 
      preprocess face for better results,
      adaptive method to analyze face
      (use facebox, overall median, face calibration phase etc...),
      analyze sight,
      code cleaning, splitting, development/production setting
*/
import * as canvas from "canvas";
import * as faceapi from "face-api.js";
import * as tfjs from "@tensorflow/tfjs";
import { status, analyze } from "./analysis.js";
tfjs.enableProdMode();

let landmarkModel = null;
let score = 0;
const video = document.getElementById("video");
const vidW = 640;
const vidH = 480;

const { ImageData } = canvas;
faceapi.env.monkeyPatch({
  Canvas: HTMLCanvasElement,
  Image: HTMLImageElement,
  ImageData: ImageData,
  Video: HTMLVideoElement,
  createCanvasElement: () => document.createElement("canvas"),
  createImageElement: () => document.createElement("img"),
});

function startVideo() {
  navigator.getUserMedia(
    { video: { width: vidW, height: vidH } },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("./models-faceapi"),
  new Promise((resolve, reject) => {
    try {
      tfjs
        .loadGraphModel("../dist/models-tfjs/keypoints_tfjs/model.json")
        .then((output) => {
          landmarkModel = output;
          resolve();
        });
    } catch (e) {
      reject(e);
    }
  }),
]).then(startVideo);

let timeLabel = 0;
video.addEventListener("play", async () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  const ctx = canvas.getContext("2d");
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  const cam = await tfjs.data.webcam(video);

  setInterval(async () => {
    if (timeLabel % 10 == 0) console.time("time fd" + timeLabel);
    const [detection, img] = await Promise.all([
      new Promise((resolve, reject) => {
        try {
          faceapi
            .detectSingleFace(
              video,
              new faceapi.TinyFaceDetectorOptions({
                inputSize: 160,
                scoreThreshold: 0.3,
              })
            )
            .then((_detection) => {
              resolve(_detection);
            });
        } catch (e) {
          reject(e);
        }
      }),
      new Promise((resolve, reject) => {
        try {
          cam.capture().then((_img) => {
            const __img = tfjs.reshape(_img, [1, vidH, vidW, 3]);
            tfjs.dispose(_img);
            resolve(__img);
          });
        } catch (e) {
          reject(e);
        }
      }),
    ]);
    if (timeLabel % 10 == 0) console.timeEnd("time fd" + timeLabel);

    if (timeLabel % 10 == 0) console.time("time lm" + timeLabel);
    if (detection) {
      const box = detection._box;
      const multX = box.width / 240;
      const multY = box.height / 240;

      const resizedImg = tfjs.image.cropAndResize(
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
      const landmarks = landmarkModel.execute(resizedImg, "Identity_2");
      landmarkConverter(landmarks, box.x, box.y, multX, multY).then(
        (landmarkObj) => {
          score = analyze(detection, landmarkObj);
          drawAll(canvas, ctx, detection, landmarkObj, displaySize, score);
        }
      );

      tfjs.dispose(landmarks);
      tfjs.dispose(resizedImg);
    } else {
      score = analyze(detection, undefined);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawInfo(ctx, score);
    }
    if (timeLabel % 10 == 0) console.timeEnd("time lm" + timeLabel);
    timeLabel += 1;

    tfjs.dispose(img);
  }, 50);
});

function drawAll(canvas, ctx, detection, landmarkObj, displaySize, score) {
  const resizedDetections = faceapi.resizeResults(detection, displaySize);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  faceapi.draw.drawDetections(canvas, resizedDetections);
  ctx.fillStyle = "#FF00FF";
  for (let i = 0; i < 68; ++i) {
    ctx.fillRect(landmarkObj[i]["_x"], landmarkObj[i]["_y"], 3, 3);
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

async function landmarkConverter(landmarks, offsetX, offsetY, multX, multY) {
  const _landmarks = await landmarks.array();
  let landmarkObj = new Array();
  for (let i = 0; i < 136; ++i) {
    if (!(i % 2)) landmarkObj[Math.floor(i / 2)] = new Object();

    let o =
      i % 2
        ? { key: "_y", offset: offsetY, mult: multY }
        : { key: "_x", offset: offsetX, mult: multX };
    landmarkObj[Math.floor(i / 2)][o.key] =
      o.offset + 240 * _landmarks[0][i] * o.mult;
  }
  return landmarkObj;
}
