// import { frames } from "./index.js";

// import { eye } from "@tensorflow/tfjs";

export let status = {
  yaw: 0,
  roll: 0,
  pitch: 0,
  detectRatio: 1,
  turned: false,
  turnedFactor: 0,
  bowed: false,
  bowedFactor: 0,
  eyesClosed: false,
  eyesClosedRatio: 0,
};

const weights = { undetected: 50, turned: 30, bowed: 40, eyesClosed: 50 };
// const eyeFactor = 5.8; // higher -> need to close eye harder to trigger true
const turnFactor = 3.5; // higher -> need more turn to trigger true
const bowFactor = -0.1; // higher -> need more bow to trigger true
// const eyeTurnCorrection = 2.5;

// var state = ["faceOn", ""];
// var frameTemp = 0;
var arrDetect = new Array(); // size : 200
var cnt_detect = 0;
var cnt_undetect = 0;
var m_detect = 0;
var m_undetect;
// export var detectRatio = 0;

const eyeSetting = document.getElementById("eyeSetting");
eyeSetting.onclick = () => (setFlag = true);
var arrSettingEye = new Array(); // size : 100
var arrEye = new Array(); // size : 200
var setFlag = false;
var eyeUser = 0.28;
var blinkUser = 0.15;

export function analyze(detection, landmarks, angle) {
  // empty seat check
  if (detection) arrDetect.push(1);
  //detect
  else arrDetect.push(0); //undetect

  cnt_detect = arrDetect.reduce((prev, curr) => prev + curr);
  cnt_undetect = arrDetect.length - cnt_detect;
  m_detect = 10 * Math.pow(cnt_detect, 3);
  m_undetect = 10 * Math.pow(cnt_undetect, 3);
  status.detectRatio = (1 * m_detect) / (m_detect + m_undetect);

  if (arrDetect.length == 200) {
    arrDetect.splice(0, 20);
  }

  if (landmarks) {
    status = { ...status, ...analyzeLandmark(landmarks) };
    status.pitch = angle[0][0].toFixed(3);
    status.yaw = angle[0][1].toFixed(3);
    status.roll = angle[0][2].toFixed(3);
  }

  // weighted sum of score to produce overall score.
  return Object.keys(weights).reduce((acc, key) => {
    const returnVal = status[key] ? acc + status[key] * weights[key] : acc;
    return returnVal;
  }, 0);
}

function calcDist(p1, p2) {
  return Math.sqrt(Math.pow(p1._x - p2._x, 2) + Math.pow(p1._y - p2._y, 2));
}

function diffBigger(l1, l2, ratio) {
  if (l1 < l2) {
    let t = l1;
    l1 = l2;
    l2 = t;
  }
  return [l2 * ratio < l1, `${(l1 / l2).toFixed(2)}>${turnFactor}`];
}

// eye blink edge threshold detect algorithm
function calcEdge(arr1) {
  var underEdge = (arr1[25] * 3) / 4 + (arr1[26] * 1) / 4;
  var overEdge = (arr1[75] * 1) / 4 + (arr1[76] * 3) / 4;
  var range = overEdge - underEdge;
  var blinkRange = underEdge - 1.5 * range;
  var arrSmall = new Array();
  var endIndex = arr1.findIndex((element) => element > blinkRange);
  arrSmall = arr1.splice(0, endIndex);
  // console.log(blinkRange, arr1[0], endIndex, arrSmall);
  return arrSmall;
}

function analyzeLandmark(landmarks) {
  // turned face
  const lcheek = calcDist(landmarks[33], landmarks[3]);
  const rcheek = calcDist(landmarks[33], landmarks[13]);
  const [turned, turnedFactor] = diffBigger(lcheek, rcheek, turnFactor);

  // bowed face
  const facehigh = (landmarks[1]._y + landmarks[15]._y) / 2;
  const eyelow = (landmarks[39]._y + landmarks[42]._y) / 2;
  const distance =
    (calcDist(landmarks[0], landmarks[1]) +
      calcDist(landmarks[15], landmarks[16])) /
    2;
  const bowed = facehigh < eyelow + distance * -bowFactor;
  const bowedFactor = `${(-(facehigh - eyelow) / distance).toFixed(
    2
  )}>${bowFactor}`;

  // eyes closed
  const r_in_h = calcDist(landmarks[38], landmarks[40]);
  const r_out_h = calcDist(landmarks[37], landmarks[41]);
  const r_w = calcDist(landmarks[36], landmarks[39]);
  const l_in_h = calcDist(landmarks[43], landmarks[47]);
  const l_out_h = calcDist(landmarks[44], landmarks[46]);
  const l_w = calcDist(landmarks[42], landmarks[45]);

  const r_eye = (r_in_h + r_out_h) / (2 * r_w);
  const l_eye = (l_in_h + l_out_h) / (2 * l_w);
  const avgEAR = (r_eye + l_eye) / 2; //average Eye Aspect Ratio

  var min_eye = blinkUser;
  var max_eye = eyeUser;
  if (setFlag) {
    if (arrSettingEye.length == 100) {
      var arrBlink = new Array();
      arrSettingEye.sort((a, b) => a - b);
      arrBlink = calcEdge(arrSettingEye);
      max_eye = arrSettingEye[-1];
      min_eye = arrBlink[0];
      eyeUser =
        arrSettingEye.reduce((a, b) => {
          return a + b;
        }, 0) / arrSettingEye.length;
      blinkUser =
        arrBlink.reduce((a, b) => {
          return a + b;
        }, 0) / arrBlink.length;
      // console.log(eyeUser, blinkUser);
      setFlag = false;
    } else {
      arrSettingEye.push(avgEAR);
      // console.log(arrSettingEye.length);
    }
  }
  // console.log(r_eye.toFixed(3), l_eye.toFixed(3), avgEAR.toFixed(3));
  arrEye.push(avgEAR);
  var avgEye =
    arrEye.reduce((a, b) => {
      return a + b;
    }, 0) / arrEye.length;

  var weight = 1;
  if (avgEye > max_eye) weight = 0.1;
  else if (avgEye < min_eye) avgEye = blinkUser;
  else weight = 1;

  const eyesClosedRatio =
    (weight * Math.abs(eyeUser - avgEye)) / (eyeUser - blinkUser);
  // if (!setFlag) console.log(status.eyesClosedRatio);
  if (arrEye.length == 200) {
    arrEye.splice(0, 20);
  }

  // const leyeHeight = calcDist(landmarks[38], landmarks[40]);
  // const reyeHeight = calcDist(landmarks[43], landmarks[47]);
  // const avgeyeHeight = (leyeHeight + reyeHeight) / 2;
  // const leyeWidth = calcDist(landmarks[36], landmarks[39]);
  // const reyeWidth = calcDist(landmarks[42], landmarks[45]);
  // const avgeyeWidth = (leyeWidth + reyeWidth) / 2;
  // const cheekRatio =
  //   Math.pow(Math.max(lcheek, rcheek) - Math.min(lcheek, rcheek), 2) /
  //   Math.pow(Math.max(lcheek, rcheek), 2);
  // const eyesClosed =
  //   avgeyeWidth / avgeyeHeight + eyeTurnCorrection * cheekRatio >= eyeFactor;
  // const eyesClosedFactor = `${(
  //   avgeyeWidth / avgeyeHeight +
  //   eyeTurnCorrection * cheekRatio
  // ).toFixed(2)} > ${eyeFactor}`;

  const eyesClosed = true;
  // const eyesClosedFactor = 3.0;
  return {
    turned,
    turnedFactor,
    bowed,
    bowedFactor,
    eyesClosed,
    eyesClosedRatio,
  };
}
