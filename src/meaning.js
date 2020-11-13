import { status } from "./lowdata.js";
import { frames } from "./index.js";

export let result = {
  absence: 0,
  arrAbsence: new Array(),
  sleep: 0,
  arrSleep: new Array(),
  turnHead: 0,
  arrTurn: new Array(),
  focusPoint: 100,
};

const stateText = document.getElementById("stateText");
const absenceCnt = document.getElementById("absenceCnt");
const sleepCnt = document.getElementById("sleepCnt");
const turnCnt = document.getElementById("turnCnt");

let state = ["check", ""];
let buffer_cnt = 0;
let threshold = { absence: 0.5, sleep: 0.8, turn: 0.18 };
let stamp = 0;
let timeAbsence, timeSleep, timeTurn;
let arrPoint = new Array();
let eyePer = 1;
let deltaPer = 1;
let mousePer = 1;

export function analyze() {
  stateText.innerHTML = "현재 상태 : " + state[0];
  absenceCnt.innerHTML =
    "부재 " +
    result.absence +
    "회" +
    ", 최근 시간 : " +
    (result.arrAbsence.length
      ? result.arrAbsence[result.arrAbsence.length - 1] + "sec"
      : "no");
  sleepCnt.innerHTML =
    "잠 " +
    result.sleep +
    "회" +
    ", 최근 시간 : " +
    (result.arrSleep.length
      ? result.arrSleep[result.arrSleep.length - 1] + "sec"
      : "no");
  turnCnt.innerHTML =
    "고개돌림 " +
    result.turnHead +
    "회" +
    ", 최근 시간 : " +
    (result.arrTurn.length
      ? result.arrTurn[result.arrTurn.length - 1] + "sec"
      : "no");
  switch (state[0]) {
    case "check":
      // console.log(state[0]);
      if (status.detectRatio < threshold.absence) state = ["buffer", "absence"];
      else if (status.eyesClosedRatio > threshold.sleep)
        state = ["buffer", "sleep"];
      else if (Math.abs(status.yaw) > threshold.turn)
        state = ["buffer", "turnHead"];
      else {
        if (status.eyeRatio < 0.15) eyePer = 0;
        else if (status.eyeRatio < 0.27)
          eyePer = 6.67 * (status.eyeRatio - 0.27) + 0.8;
        //0.8
        else if (status.eyeRatio < 0.34) eyePer = 0.8;
        else if (status.eyeRatio < 0.4)
          eyePer = 3.34 * (status.eyeRatio - 0.4) + 1;
        else eyePer = 1;

        if (status.deltaEyeRatio < 0) deltaPer = 0;
        else if (status.deltaEyeRatio < 0.125)
          deltaPer = 6.4 * (status.deltaEyeRatio - 0.125) + 0.8;
        else if (status.deltaEyeRatio < 0.1875) deltaPer = 0.8;
        else if (status.deltaEyeRatio < 0.258)
          deltaPer = 2.82 * (status.deltaEyeRatio - 0.258) + 1;
        else deltaPer = 1;

        if (status.mouseRatio < 0.5) mousePer = 1;
        else if (status.mouseRatio < 0.8)
          mousePer = -5.56 * Math.pow(status.mouseRatio - 0.5, 2) + 1;
        else if (status.mouseRatio < 1.2)
          mousePer = 3.125 * Math.pow(status.mouseRatio - 1.2, 2);
        else mousePer = 0;

        arrPoint.push(((eyePer * 2 + deltaPer * 1 + mousePer * 7) * 100) / 10);
        if (arrPoint.length % 10 == 0) {
          result.focusPoint = (
            arrPoint.reduce((a, b) => {
              return a + b;
            }, 0) / arrPoint.length
          ).toFixed(1);
        }
        if (arrPoint.length == 100) arrPoint.splice(0, 20);
        // console.log(result.focusPoint);
      }
      break;
    case "absence":
      timeAbsence = frames - stamp;
      if (timeAbsence * 0.05 > 5) {
        result.absence = result.arrAbsence.length + 1;
        if (status.detectRatio > threshold.absence) {
          result.arrAbsence.push((timeAbsence * 0.05).toFixed(2));
          state[0] = "check";
        }
      }
      break;
    case "sleep":
      timeSleep = frames - stamp;
      if (timeSleep * 0.05 > 5) {
        result.sleep = result.arrSleep.length + 1;
        if (
          status.eyesClosedRatio < threshold.sleep ||
          status.detectRatio < threshold.absence
        ) {
          result.arrSleep.push((timeSleep * 0.05).toFixed(2));
          state[0] = "check";
        }
      }
      break;
    case "turnHead":
      timeTurn = frames - stamp;
      if (timeTurn * 0.05 > 4) {
        result.turnHead = result.arrTurn.length + 1;
        if (
          Math.abs(status.yaw) < threshold.turn ||
          status.detectRatio < threshold.absence
        ) {
          result.arrTurn.push((timeTurn * 0.05).toFixed(2));
          state[0] = "check";
        }
      }
      break;
    case "buffer":
      if (state[1] == "absence" && status.detectRatio > threshold.absence) {
        buffer_cnt = 0;
        state[0] = "check";
      } else if (
        state[1] == "sleep" &&
        status.eyesClosedRatio < threshold.sleep
      ) {
        buffer_cnt = 0;
        state[0] = "check";
      } else if (
        state[1] == "turnHead" &&
        Math.abs(status.yaw) < threshold.turn
      ) {
        buffer_cnt = 0;
        state[0] = "check";
      } else if (buffer_cnt > 20) {
        buffer_cnt = 0;
        stamp = frames;
        state[0] = state[1];
      } else {
        buffer_cnt++;
      }
      break;
  }
  return result;
}
