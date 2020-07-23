let prevExpression = "neutral";

export const status = {
  turned: false,
  bowed: false,
  expressionChange: false,
};

export function analyze(detections) {
  if (detections[0]) {
    let landmarks = detections[0].landmarks._positions;
    [status.turned, status.bowed] = analyzeLandmark(landmarks);

    let expressions = detections[0].expressions;
    analyzeExpression(expressions);
  }
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
  return l2 < l1 * ratio;
}

function analyzeLandmark(landmarks) {
  let lcheek = calcDist(landmarks[33], landmarks[3]);
  let rcheek = calcDist(landmarks[33], landmarks[13]);
  let turned = diffBigger(lcheek, rcheek, 0.4);

  let facehigh = (landmarks[1]._y + landmarks[15]._y) / 2;
  let eyelow = (landmarks[39]._y + landmarks[42]._y) / 2;
  let bowed = facehigh < eyelow;

  return [turned, bowed];
}

function analyzeExpression(expressions) {
  let curExpression = "neutral";
  Object.keys(expressions).forEach((key) => {
    if (expressions[curExpression] < expressions[key]) curExpression = key;
  });
  if (prevExpression != curExpression && status.expressionChange != true) {
    status.expressionChange = true;
    setTimeout(
      (status) => {
        status.expressionChange = false;
      },
      1500,
      status
    );
  }
}
