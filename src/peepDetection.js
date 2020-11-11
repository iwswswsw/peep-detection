import {parse} from 'querystring';
import * as posenet from '@tensorflow-models/posenet';
import * as blazeface from '@tensorflow-models/blazeface';
import Stats from 'stats.js';
import {
  drawKeypoints, drawSkeleton, isMobile, toPointObject, distancePoints,
} from './utils';
import {setupCamera} from './setupCamera';
import {colors} from './const';
import sunglassURL from './img/sunglass_normal.png';

const stats = new Stats();

let sunglassImg;

const loadImage = (src) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = src;
  });
};

const loadVideo = async () => {
  const video = await setupCamera();
  video.play();

  return video;
};

const setupFPS = () => {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
};

const detectPoseInRealTime = (video, net, canvas) => {
  const ctx = canvas.getContext('2d');
  const minPoseConfidence = 0.15;
  const minPartConfidence = 0.1;

  const poseDetectionFrame = async () => {
    stats.begin();
    let poses = [];
    let allPoses= await net.estimatePoses(
      video, {
        decodingMethod: 'multi-person',
        // imageScaleFactor: 0.2,
        flipHorizontal: true, // since images are being fed from a webcam
        maxDetections: 5,
        scoreThreshold: minPartConfidence,
        nmsRadius: 30.0,
      }
    );
    poses = poses.concat(allPoses);
    const posesConf = poses.filter(({score}) => score >= minPoseConfidence);

    if (posesConf.length > 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
      ctx.restore();

      // 中心にある顔をメイン利用者としてフラグを立てる
      const canvasCenter = {
        x: canvas.width / 2,
        y: canvas.height / 2,
      };
      const mainPose = posesConf.reduce((prev, curr) => {
        // （landmarks[3]は鼻の座標）
        return distancePoints(curr.keypoints[0].position, canvasCenter) < distancePoints(prev.keypoints[0].position, canvasCenter)
          ? curr : prev;
      });
      mainPose.main = true;

      // プライバシーモードの取得
      const isPrivacyModeOn = document.getElementById('privacy-mode').privacy.value === 'on';

      posesConf.forEach(({score, keypoints, main}, i) => {
        // メイン利用者以外は違う色で描画する
        drawKeypoints(keypoints, minPartConfidence, ctx, colors[main ? 0 : 1]);
        drawSkeleton(keypoints, minPartConfidence, ctx, colors[main ? 0 : 1]);
        // drawBoundingBox(keypoints, ctx);

        if (isPrivacyModeOn) {
          // 目線を入れる
          // drawEyeLine(keypoints, minPartConfidence, ctx);

          // サングラスを描く
          const leyePoint = keypoints[1].position;
          const reyePoint = keypoints[2].position;
          const scale = (reyePoint.x - leyePoint.x) / 110;
          const sw = sunglassImg.width * scale;
          const sh = sunglassImg.height * scale;
          const center = {
            x: (reyePoint.x + leyePoint.x) / 2,
            y: (reyePoint.y + leyePoint.y) / 2,
          };
          const angle = Math.atan2(reyePoint.y - leyePoint.y, reyePoint.x - leyePoint.x);
          const drawPoint = {
            x: - sw / 2,
            y: - sh / 2.3, // 中心より少し下に描画する
          };

          ctx.save();
          ctx.translate(center.x, center.y);
          ctx.rotate(angle);
          ctx.drawImage(sunglassImg, drawPoint.x, drawPoint.y, sw, sh);
          ctx.restore();
        }
      });
    }

    stats.end();

    // 認識しているポーズ数の表示
    const text = `poses count (show / all): ${posesConf.length} / ${poses.length} ${posesConf.length > 1 ? '覗き込みを検知しています（赤点・赤線）' : ''}`;
    const textNode = document.createTextNode(text);
    const newElem = document.createElement('p').appendChild(textNode);
    const parent = document.getElementById('raw-data');
    parent.replaceChild(newElem, parent.childNodes[0]);

    requestAnimationFrame(poseDetectionFrame);
  };

  poseDetectionFrame();
};

const detectFaceInRealTime = (video, model, canvas) => {
  const ctx = canvas.getContext('2d');
  const minConfidence = 0.15;

  const renderPrediction = async () => {
    stats.begin();

    const returnTensors = false;
    const flipHorizontal = true;
    const annotateBoxes = true;
    const predictions = await model.estimateFaces(video, returnTensors, flipHorizontal, annotateBoxes);

    const predConf = predictions.filter(({probability}) => probability >= minConfidence);

    // プライバシーモードの取得
    // TODO: posenet側との共通化
    const isPrivacyModeOn = document.getElementById('privacy-mode').privacy.value === 'on';

    if (predConf.length > 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
      ctx.restore();

      // 中心にある顔をメイン利用者としてフラグを立てる
      const canvasCenter = {
        x: canvas.width / 2,
        y: canvas.height / 2,
      };
      const mainPred = predConf.reduce((prev, curr) => {
        // （landmarks[3]は鼻の座標）
        return distancePoints(toPointObject(curr.landmarks[3]), canvasCenter) < distancePoints(toPointObject(prev.landmarks[3]), canvasCenter)
          ? curr : prev;
      });
      mainPred.main = true;

      for (let i = 0; i < predConf.length; i++) {
        if (returnTensors) {
          predConf[i].topLeft = predConf[i].topLeft.arraySync();
          predConf[i].bottomRight = predConf[i].bottomRight.arraySync();
          if (annotateBoxes) {
            predConf[i].landmarks = predConf[i].landmarks.arraySync();
          }
        }

        // メイン利用者以外の顔全体に赤い四角を描く
        if (!predConf[i].main) {
          const start = predConf[i].topLeft;
          const end = predConf[i].bottomRight;
          const size = [end[0] - start[0], end[1] - start[1]];
          ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
          ctx.fillRect(start[0], start[1], size[0], size[1]);
        }

        const landmarks = predConf[i].landmarks;
        if (annotateBoxes) {
          ctx.fillStyle = 'blue';
          for (let j = 0; j < landmarks.length; j++) {
            const x = landmarks[j][0];
            const y = landmarks[j][1];
            ctx.fillRect(x, y, 5, 5);
          }
        }

        if (isPrivacyModeOn) {
          // 目線を描く
          // posenet側と同じ関数使いたい
          // const keypointLeftEye = landmarks[1]; // [x, y]
          // const keypointRightEye = landmarks[0];
          // const keypointLeftEar = landmarks[5];
          // const keypointRightEar = landmarks[4];

          // ctx.beginPath();
          // ctx.moveTo(keypointLeftEar[0], keypointLeftEye[1]);
          // ctx.lineTo(keypointRightEar[0], keypointRightEye[1]);
          // ctx.lineWidth = 100;
          // ctx.strokeStyle = 'black';
          // ctx.stroke();

          // サングラスを描く
          const [leyeX, leyeY] = landmarks[1];
          const [reyeX, reyeY] = landmarks[0];
          const scale = (reyeX - leyeX) / 130; // posenetよりも少し大きめに描画されてしまうので、スケール調整
          const sw = sunglassImg.width * scale;
          const sh = sunglassImg.height * scale;
          const center = {
            x: (reyeX + leyeX) / 2,
            y: (reyeY + leyeY) / 2,
          };
          const angle = Math.atan2(reyeY - leyeY, reyeX - leyeX);
          const drawPoint = {
            x: - sw / 2,
            y: - sh / 2.3, // 中心より少し下に描画する
          };

          ctx.save();
          ctx.translate(center.x, center.y);
          ctx.rotate(angle);
          ctx.drawImage(sunglassImg, drawPoint.x, drawPoint.y, sw, sh);
          ctx.restore();
        }
      }
    }

    stats.end();

    // 認識しているポーズ数の表示
    // TODO: posenet側との共通化
    const text = `poses count (show / all): ${predConf.length} / ${predictions.length} ${predConf.length > 1 ? '覗き込みを検知しています（赤四角）' : ''}`;
    const textNode = document.createTextNode(text);
    const newElem = document.createElement('p').appendChild(textNode);
    const parent = document.getElementById('raw-data');
    parent.replaceChild(newElem, parent.childNodes[0]);

    requestAnimationFrame(renderPrediction);
  };

  renderPrediction();
};

const bindPage = async () => {
  // 利用モデルの取得
  const params = parse(location.search.slice(1)); // 最初の'?'を除去しつつ
  const isPosenetUsed = params.model === 'posenet' || Object.keys(params).length === 0;

  // 利用モデルの表示
  const usedModelName = isPosenetUsed ? 'posenet' : 'blazeface';
  const text = document.createTextNode(`利用モデル: ${usedModelName}`);
  const newElem = document.createElement('p').appendChild(text);
  const parent = document.getElementById('model-mode');
  parent.replaceChild(newElem, parent.childNodes[0]);

  let model;
  let video;

  // モデルの呼び出し
  if (isPosenetUsed) {
    model = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: 500,
      multiplier: isMobile() ? 0.50 : 0.75,
      quantBytes: 2,
    });
  } else {
    model = await blazeface.load();
  }

  // video属性の呼び出し
  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  video.width = video.videoWidth;
  video.height = video.videoHeight;

  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // リソース利用状況表示窓の呼び出し
  setupFPS();

  sunglassImg = await loadImage(sunglassURL);

  // モデルの予測を表示
  if (isPosenetUsed) {
    detectPoseInRealTime(video, model, canvas);
  } else {
    detectFaceInRealTime(video, model, canvas);
  }
};

bindPage();
