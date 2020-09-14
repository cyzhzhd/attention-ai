export let status = {
  yaw: 0,
  roll: 0,
  pitch: 0,
  detectRatio: 1,
  eyeRatio: 0,
  deltaEyeRatio: 0,
  mouseRatio: 0,
  eyesClosedRatio: 0,
};

// detect part variables
var arrDetect = new Array(); // size : 200
var cnt_detect = 0; //count
var cnt_undetect = 0; //count
var m_detect = 0; //weight
var m_undetect; //weight

// eye close part variables
const eyeSetting = document.getElementById("eyeSetting");
const sec_cnt = document.getElementById("settingSec");
eyeSetting.onclick = () => (setFlag = true);
var arrSettingEye = new Array(); // size : 100, buffer to set eye size
var arrEye = new Array(); // size : 200
var setFlag = false; // user eye setting mode
var eyeUser = 0.28; // initial value
var blinkUser = 0.15; // initial value

// eye height part variables
var delta_cnt = 0; // end : 10
var delta_eye_ratio = 0;
var temp_eye_height = 0;

//
var arrPitch = new Array();
var arrYaw = new Array();
var arrRoll = new Array();

export function rowData(detection, landmarks, angle) {
  // empty seat check
  if (detection) arrDetect.push(1);
  //detect
  else arrDetect.push(0); //undetect

  cnt_detect = arrDetect.reduce((prev, curr) => prev + curr);
  cnt_undetect = arrDetect.length - cnt_detect;
  m_detect = 10 * Math.pow(cnt_detect, 3);
  m_undetect = 10 * Math.pow(cnt_undetect, 3);
  status.detectRatio = ((1 * m_detect) / (m_detect + m_undetect)).toFixed(3);

  if (arrDetect.length == 200) {
    arrDetect.splice(0, 20);
  }

  if (landmarks) {
    status = { ...status, ...analyzeLandmark(landmarks) };
    arrPitch.push(Number(angle[0][0].toFixed(3)));
    arrYaw.push(Number(angle[0][1].toFixed(3)));
    arrRoll.push(Number(angle[0][2].toFixed(3)));
    // console.log(arrYaw);
    if (arrPitch.length == 100) arrPitch.splice(0, 20);
    if (arrYaw.length == 100) arrYaw.splice(0, 20);
    if (arrRoll.length == 100) arrRoll.splice(0, 20);
    status.pitch = (
      arrPitch.reduce((a, b) => {
        return a + b;
      }, 0) / arrPitch.length
    ).toFixed(3);
    status.yaw = (
      arrYaw.reduce((a, b) => {
        return a + b;
      }, 0) / arrYaw.length
    ).toFixed(3);
    status.roll = (
      arrRoll.reduce((a, b) => {
        return a + b;
      }, 0) / arrRoll.length
    ).toFixed(3);
  }

  // weighted sum of score to produce overall score.
  return status;
}

function calcDist(p1, p2) {
  return Math.sqrt(Math.pow(p1._x - p2._x, 2) + Math.pow(p1._y - p2._y, 2));
}

// function diffBigger(l1, l2, ratio) {
//   if (l1 < l2) {
//     let t = l1;
//     l1 = l2;
//     l2 = t;
//   }
//   return [l2 * ratio < l1, `${(l1 / l2).toFixed(2)}>${turnFactor}`];
// }

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
  // eyes closed
  const r_in_h = calcDist(landmarks[38], landmarks[40]);
  const r_out_h = calcDist(landmarks[37], landmarks[41]);
  const r_w = calcDist(landmarks[36], landmarks[39]);
  const l_in_h = calcDist(landmarks[43], landmarks[47]);
  const l_out_h = calcDist(landmarks[44], landmarks[46]);
  const l_w = calcDist(landmarks[42], landmarks[45]);

  const mouse_w = calcDist(landmarks[48], landmarks[54]);
  const mouse_h =
    (calcDist(landmarks[50], landmarks[58]) +
      calcDist(landmarks[51], landmarks[57]) +
      calcDist(landmarks[52], landmarks[56])) /
    3;
  const mouse_ratio = mouse_h / mouse_w;

  const r_eye_height = (r_in_h + r_out_h) / 2;
  const l_eye_height = (l_in_h + l_out_h) / 2;
  // const eye_height = (r_eye_height + l_eye_height) / 2;

  const r_eye = r_eye_height / r_w;
  const l_eye = l_eye_height / l_w;
  const avgEAR = (r_eye + l_eye) / 2; //average Eye Aspect Ratio

  delta_cnt++;
  if (delta_cnt == 20) {
    // 1sec
    delta_cnt = 0;
    delta_eye_ratio = Math.abs(avgEAR - temp_eye_height);
    temp_eye_height = avgEAR;
  }

  var min_eye = blinkUser;
  var max_eye = eyeUser;

  if (setFlag) {
    sec_cnt.innerHTML =
      5 -
      parseInt(arrSettingEye.length / 20) +
      "초 동안 화면을 정면으로 응시해주세요.";
    if (arrSettingEye.length == 100) {
      sec_cnt.innerHTML = "사용자 눈 크기 조정이 완료되었습니다.";
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
      arrSettingEye = new Array();
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
  else weight = 1 - 20 * Math.pow(status.pitch, 2);
  if (status.pitch < 0 || status.pitch > 0.18) weight = 0.1;

  const eyesClosedRatio = (
    (weight * Math.abs(eyeUser - avgEye)) /
    (eyeUser - blinkUser)
  ).toFixed(3);
  // if (!setFlag) console.log(status.eyesClosedRatio);
  if (arrEye.length == 200) {
    arrEye.splice(0, 20);
  }

  const eyeRatio = avgEAR.toFixed(3);
  const deltaEyeRatio = delta_eye_ratio.toFixed(3);
  const mouseRatio = mouse_ratio.toFixed(3);

  return {
    eyeRatio,
    deltaEyeRatio,
    mouseRatio,
    eyesClosedRatio,
  };
}
