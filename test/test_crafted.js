// this script only works on recent tfjs-node version
const tfjs = require("@tensorflow/tfjs-node");
const path = require("path");

async function main() {
  let craftedModel = null;
  craftedModel = await tfjs.loadGraphModel(
    `file://${path.join("../dist/models-tfjs/detector_crafted/model.json")}`
  );

  for (let i = 0; i < 10; i++) {
    console.time("inference");
    await craftedModel.predict(tfjs.ones([1, 128, 128, 3])).print();
    console.timeEnd("inference");
  }
}

main();
