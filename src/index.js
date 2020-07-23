import * as canvas from "canvas";
import * as faceapi from "face-api.js";
import { status, analyze } from "./analysis.js";

const weights = { turned: 30, bowed: 40, expressionChange: 20 };

const { ImageData } = canvas;
faceapi.env.monkeyPatch({
  Canvas: HTMLCanvasElement,
  Image: HTMLImageElement,
  ImageData: ImageData,
  Video: HTMLVideoElement,
  createCanvasElement: () => document.createElement("canvas"),
  createImageElement: () => document.createElement("img"),
});
const video = document.getElementById("video");

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("./models-faceapi"),
  faceapi.nets.faceLandmark68Net.loadFromUri("./models-faceapi"),
  faceapi.nets.faceRecognitionNet.loadFromUri("./models-faceapi"),
  faceapi.nets.faceExpressionNet.loadFromUri("./models-faceapi"),
]).then(startVideo);

let timeLabel = 0;
video.addEventListener("play", () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    console.time("time" + timeLabel);
    const detections = await faceapi
      .detectAllFaces(
        video,
        new faceapi.TinyFaceDetectorOptions({
          inputSize: 320,
          scoreThreshold: 0.4,
        })
      )
      .withFaceLandmarks()
      .withFaceExpressions();
    console.timeEnd("time" + timeLabel);
    timeLabel += 1;

    analyze(detections);
    // console.log(status);
    let score = Object.keys(weights).reduce((acc, key) => {
      return acc + status[key] * weights[key];
    }, 0);

    let ctx = canvas.getContext("2d");
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
    ctx.font = "30px Arial";
    ctx.fillStyle = "#000000";
    ctx.fillText("score: " + score, 30, 50);
  }, 100);
});
