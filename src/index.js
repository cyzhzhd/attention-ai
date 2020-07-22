import * as canvas from "canvas";
import * as faceapi from "face-api.js";
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
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
  }, 100);
});
