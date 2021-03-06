/*
TODO: 
      constants to another file
      benchmark
      adaptive method to analyze face
      (overall median, face calibration phase etc...),
      analyze sight,
      development/production setting
*/
import * as tfjs from "@tensorflow/tfjs";
// import { status, rowData } from "./lowdata.js";
import { rowData } from "./lowdata.js";
import { landmarkModel } from "./landmark.js";
import { detectorModel } from "./detector.js";
import { analyze, result } from "./meaning.js";
tfjs.enableProdMode();

export let frames = 0;
const stop = document.getElementById("stopButton");
const resume = document.getElementById("resumeButton");
const video = document.getElementById("video");
console.log(video);

const vidW = 640;
const vidH = 480;

stop.onclick = onStop;
resume.onclick = onResume;

let task = true;
function onStop() {
  task = false;
  console.log(video);
  video.pause();
}
function onResume() {
  task = true;
  video.load();
}

Promise.all([
  landmarkModel.loadFromUri("../dist/models-tfjs/keypoints_tfjs/model.json"),
  detectorModel.loadFromUri(
    "../dist/models-tfjs/detector_crafted_q/model.json"
  ),
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
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  document.body.append(canvas);
  setInterval(async function faceAnalysis() {
    if (!task) return;

    // const timefd1 = performance.now();
    const pixel = tfjs.browser.fromPixels(video);
    // console.log(pixel);
    const img = pixel.reshape([-1, vidH, vidW, 3]);
    const detectImg = tfjs.image.resizeBilinear(img, [128, 128]);
    const [bbox, conf] = await detectorModel.predict(detectImg);
    // const timefd2 = performance.now();

    // const timelm1 = performance.now();
    const [angle, landmark] = await landmarkModel.predict(bbox, img);
    // const timelm2 = performance.now();

    JSON.stringify(rowData(bbox, landmark, angle), null, 2);
    analyze();
    drawAll(canvas, ctx, bbox, conf, landmark, result);

    // time checker
    frames = frames + 1;
    if (frames % 10 === 0) {
      // console.log(
      //   `${frames}: fd ${(timefd2 - timefd1).toFixed(3)}ms lm ${(
      //     timelm2 - timelm1
      //   ).toFixed(3)}ms`
      // );
      // console.log(status.detectRatio, status.eyesClosedRatio);
    }
    tfjs.dispose([landmark, detectImg, angle, pixel, img]);
  }, 50);
});

function drawAll(canvas, ctx, bbox, conf, landmarkObj, result1) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#FF0000";
  ctx.strokeStyle = "#FF0000";
  ctx.font = "30px Arial";
  ctx.lineWidth = "4";

  if (bbox !== undefined) {
    ctx.font = "30px Arial";
    ctx.beginPath();
    ctx.rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    ctx.fillText(conf.toFixed(2), bbox[0] + 15, bbox[1] + 30);
    ctx.stroke();
    for (let i = 0; i < 68; ++i) {
      ctx.fillRect(landmarkObj[i]["_x"], landmarkObj[i]["_y"], 4, 4);
    }
  }
  drawInfo(ctx, result1);
}

function drawInfo(ctx, result2) {
  ctx.fillText("score: " + result2.focusPoint, 10, 40);
  ctx.font = "12px Arial";
  // var lines = JSON.stringify(status, null, 2).split("\n");
  // var lines = low_data.split("\n");
  ctx.fillText("absense count : " + result2.absence, 10, 240 + 1 * 20);
  ctx.fillText("sleep count : " + result2.sleep, 10, 240 + 2 * 20);
  ctx.fillText("turnHead count : " + result2.turnHead, 10, 240 + 3 * 20);
  ctx.font = "14px Arial";
  ctx.fillText(JSON.stringify(tfjs.memory()), 20, 470);
}
