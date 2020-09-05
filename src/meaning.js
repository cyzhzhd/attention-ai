//   switch (state[0]) {
//     case "faceOn":
//       if (status.undetected) {
//         console.log("undetect start");
//         console.log(frames);
//         frameTemp = frames;
//         state[0] = "undetectAcc";
//       }
//       break;

//     case "undetectAcc":
//       // console.log(frameTemp);
//       if (!status.undetected) {
//         console.log(frames);
//         frameTemp = frames;
//         state = ["detectAcc", "undetectAcc"];
//       } else if (frames - frameTemp > 40) {
//         console.log("go stepOut");
//         state[0] = "stepOut";
//       } else {
//         console.log("2 sec...");
//       }
//       break;

//     case "stepOut":
//       if (!status.undetected) {
//         console.log(frames);
//         frameTemp = frames;
//         state = ["detectAcc", "stepOut"];
//       } else console.log("He is sleeping!!!");
//       break;

//     case "detectAcc":
//       if (status.undetected) {
//         state = [state[1], ""];
//       } else if (frames - frameTemp > 20) {
//         state = ["faceOn", ""];
//       } else console.log("1 sec...");
//       break;

//     default:
//       console.error("state overflow");
//   }
