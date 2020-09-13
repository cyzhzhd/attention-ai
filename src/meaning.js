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
let threshold = { absence: 0.5, sleep: 0.5, turn: 0.18 };
let stamp = 0;
let timeAbsence, timeSleep, timeTurn;

export function analyze() {
  stateText.innerHTML = state[0];
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
      else state[0] = "check"; //focus
      break;
    case "absence":
      timeAbsence = frames - stamp;
      if (status.detectRatio > threshold.absence) {
        if (timeAbsence * 0.05 > 5) {
          result.arrAbsence.push((timeAbsence * 0.05).toFixed(2));
          result.absence = result.arrAbsence.length;
        }
        state[0] = "check";
      }
      break;
    case "sleep":
      timeSleep = frames - stamp;
      if (status.eyesClosedRatio < threshold.sleep) {
        if (timeSleep * 0.05 > 5) {
          result.arrSleep.push((timeSleep * 0.05).toFixed(2));
          result.sleep = result.arrSleep.length;
        }
        state[0] = "check";
      }
      break;
    case "turnHead":
      timeTurn = frames - stamp;
      if (
        Math.abs(status.yaw) < threshold.turn ||
        status.detectRatio < threshold.absence
      ) {
        if (timeTurn * 0.05 > 4) {
          result.arrTurn.push((timeTurn * 0.05).toFixed(2));
          result.turnHead = result.arrTurn.length;
        }
        state[0] = "check";
      }
      break;
    case "focus":
      // console.log(state[0]);
      //
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
}
