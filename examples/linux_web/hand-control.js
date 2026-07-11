// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

"use strict";

const MEDIAPIPE_VERSION = "0.10.35";
const MEDIAPIPE_ROOT =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/" +
  "gesture_recognizer.task";
const DETECTION_INTERVAL_MS = 1000 / 15;
const SMOOTHING_ALPHA = 0.28;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const FINGER_JOINTS = {
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};
const CANNED_GESTURES = {
  Victory: "victory",
  Open_Palm: "open_palm",
  Closed_Fist: "closed_fist",
};
const HAND_COLORS = ["#78f2be", "#b39cff"];

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

function cosineAt(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
  const denominator = Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z);
  if (denominator < 1e-6) return 1;
  return (ab.x * cb.x + ab.y * cb.y + ab.z * cb.z) / denominator;
}

function fingerExtensionScore(landmarks, joints, palmScale) {
  const [mcp, pip, , tip] = joints;
  const straightness = clamp01((-cosineAt(
    landmarks[mcp], landmarks[pip], landmarks[tip],
  ) - 0.45) / 0.5);
  const reachDelta = (
    distance(landmarks[tip], landmarks[0])
    - distance(landmarks[pip], landmarks[0])
  ) / palmScale;
  const reach = clamp01((reachDelta - 0.03) / 0.32);
  return 0.78 * straightness + 0.22 * reach;
}

export function pinchStrength(landmarks, invert = false) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;
  const palmScale = distance(landmarks[0], landmarks[9]);
  if (!Number.isFinite(palmScale) || palmScale < 1e-5) return null;
  const pinchRatio = distance(landmarks[4], landmarks[8]) / palmScale;
  const value = clamp01((pinchRatio - 0.18) / (1.25 - 0.18));
  return invert ? 1 - value : value;
}

export function foxGestureScore(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return 0;
  const palmScale = distance(landmarks[0], landmarks[9]);
  if (!Number.isFinite(palmScale) || palmScale < 1e-5) return 0;
  const extension = Object.fromEntries(
    Object.entries(FINGER_JOINTS).map(([name, joints]) => [
      name,
      fingerExtensionScore(landmarks, joints, palmScale),
    ]),
  );
  const thumbContact = (
    distance(landmarks[4], landmarks[12])
    + distance(landmarks[4], landmarks[16])
  ) / (2 * palmScale);
  const contactScore = 1 - clamp01((thumbContact - 0.18) / 0.5);
  return Math.min(
    extension.index,
    extension.pinky,
    1 - extension.middle,
    1 - extension.ring,
    contactScore,
  );
}

export function smoothStrength(previous, next, alpha = SMOOTHING_ALPHA) {
  if (previous === null || !Number.isFinite(previous)) return next;
  return previous + alpha * (next - previous);
}

function recognizedGesture(landmarks, cannedCategory) {
  const foxScore = foxGestureScore(landmarks);
  if (foxScore >= 0.58) return { name: "fox", confidence: foxScore };
  const name = CANNED_GESTURES[cannedCategory?.categoryName] ?? null;
  if (!name) return { name: null, confidence: 0 };
  return { name, confidence: cannedCategory.score ?? 0 };
}

export class HandPromptController {
  constructor({ video, canvas, getInvert, onFrame, onStatus }) {
    this.video = video;
    this.canvas = canvas;
    this.getInvert = getInvert;
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.recognizer = null;
    this.stream = null;
    this.active = false;
    this.animationFrame = null;
    this.lastDetectionMs = -Infinity;
    this.lastVideoTime = -1;
    this.smoothedPinch = new Map();
  }

  async start() {
    if (this.active) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "カメラAPIを利用できません。SSHトンネル経由の127.0.0.1をChromeで開いてください。",
      );
    }
    this.onStatus("loading", "MediaPipe Gesture Recognizerを読み込んでいます…");
    if (!this.recognizer) this.recognizer = await this.#createRecognizer();

    this.onStatus("loading", "カメラの許可を待っています…");
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;
    this.active = true;
    this.lastDetectionMs = -Infinity;
    this.lastVideoTime = -1;
    this.smoothedPinch.clear();
    this.onStatus("searching", "両手を認識できます。手をカメラに映してください。");
    this.animationFrame = requestAnimationFrame((now) => this.#render(now));
  }

  stop(notify = true) {
    this.active = false;
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.pause();
    this.video.srcObject = null;
    this.canvas.getContext("2d")?.clearRect(
      0, 0, this.canvas.width, this.canvas.height,
    );
    if (notify) this.onStatus("idle", "カメラ停止中");
  }

  async #createRecognizer() {
    const { FilesetResolver, GestureRecognizer } = await import(
      `${MEDIAPIPE_ROOT}/vision_bundle.mjs`
    );
    const vision = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_ROOT}/wasm`);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
      cannedGesturesClassifierOptions: {
        maxResults: 1,
        scoreThreshold: 0.45,
      },
    });
    try {
      return await GestureRecognizer.createFromOptions(vision, options("GPU"));
    } catch (gpuError) {
      console.warn("MediaPipe GPU delegate failed; using CPU", gpuError);
      return GestureRecognizer.createFromOptions(vision, options("CPU"));
    }
  }

  #render(now) {
    if (!this.active) return;
    this.animationFrame = requestAnimationFrame((next) => this.#render(next));
    if (now - this.lastDetectionMs < DETECTION_INTERVAL_MS) return;
    if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (this.video.currentTime === this.lastVideoTime) return;
    this.lastDetectionMs = now;
    this.lastVideoTime = this.video.currentTime;

    try {
      const result = this.recognizer.recognizeForVideo(this.video, now);
      const handedness = result.handedness ?? result.handednesses ?? [];
      const hands = (result.landmarks ?? []).map((landmarks, index) => {
        const handCategory = handedness[index]?.[0];
        const handName = handCategory?.categoryName || `Hand ${index + 1}`;
        const gesture = recognizedGesture(landmarks, result.gestures?.[index]?.[0]);
        const rawPinch = pinchStrength(landmarks, Boolean(this.getInvert()));
        const previous = this.smoothedPinch.get(handName) ?? null;
        const smoothed = rawPinch === null
          ? previous
          : smoothStrength(previous, rawPinch);
        if (smoothed !== null) this.smoothedPinch.set(handName, smoothed);
        return {
          landmarks,
          handedness: handName,
          handednessConfidence: handCategory?.score ?? 0,
          gesture: gesture.name,
          gestureConfidence: gesture.confidence,
          pinchStrength: smoothed,
        };
      });
      this.#draw(hands);
      if (hands.length === 0) {
        this.onFrame({ hands: [] });
        this.onStatus("searching", "手を探索中（最後の強度を保持）");
        return;
      }
      this.onStatus("tracking", `${hands.length}手を追跡中`);
      this.onFrame({ hands });
    } catch (error) {
      console.error("Hand gesture recognition failed", error);
      this.onStatus("error", `手追跡エラー: ${error.message}`);
      this.stop(false);
    }
  }

  #draw(hands) {
    const context = this.canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = Math.max(2, this.canvas.width / 240);

    hands.forEach((hand, handIndex) => {
      const color = HAND_COLORS[handIndex % HAND_COLORS.length];
      const point = (index) => ({
        x: hand.landmarks[index].x * this.canvas.width,
        y: hand.landmarks[index].y * this.canvas.height,
      });
      context.strokeStyle = color;
      for (const [start, end] of HAND_CONNECTIONS) {
        const a = point(start);
        const b = point(end);
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.stroke();
      }
      hand.landmarks.forEach((_, index) => {
        const p = point(index);
        const isPinchTip = index === 4 || index === 8;
        context.beginPath();
        context.arc(p.x, p.y, isPinchTip ? 7 : 4, 0, Math.PI * 2);
        context.fillStyle = isPinchTip ? "#ffcf72" : color;
        context.fill();
      });
    });
  }
}
