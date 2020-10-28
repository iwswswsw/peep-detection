import * as posenet from '@tensorflow-models/posenet';
import Stats from 'stats.js';
import {drawKeypoints, drawSkeleton, isMobile} from './utils';
import {setupCamera} from './setupCamera';
import {videoWidth, videoHeight, colors} from './const';

const stats = new Stats();

const loadVideo = async () => {
  const video = await setupCamera();
  video.play();

  return video;
};

const setupFPS = () => {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
};

const detectPoseInRealTime = (video, net) => {
  const canvas = document.getElementById('canvas');
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

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-videoWidth, 0);
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    ctx.restore();

    posesConf.forEach(({score, keypoints}, i) => {
      drawKeypoints(keypoints, minPartConfidence, ctx, colors[i % colors.length]);
      drawSkeleton(keypoints, minPartConfidence, ctx, colors[i % colors.length]);
      // drawBoundingBox(keypoints, ctx);
    });

    stats.end();

    const text = document.createTextNode(`poses count (show / all): ${posesConf.length} / ${poses.length}`);
    const newElem = document.createElement('p').appendChild(text);
    const parent = document.getElementById('raw-data');
    parent.replaceChild(newElem, parent.childNodes[0]);

    requestAnimationFrame(poseDetectionFrame);
  };

  poseDetectionFrame();
};

const bindPage = async () => {
  const net = await posenet.load({ // posenetの呼び出し
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: 500,
    multiplier: isMobile() ? 0.50 : 0.75,
    quantBytes: 2,
  });
  let video;
  try {
    video = await loadVideo(); // video属性をロード
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupFPS();
  detectPoseInRealTime(video, net);
};

bindPage();
