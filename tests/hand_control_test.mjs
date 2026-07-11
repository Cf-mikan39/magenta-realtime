// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {
  foxGestureScore,
  pinchStrength,
  smoothStrength,
} from '../examples/linux_web/hand-control.js';

function landmarksWithPinchDistance(distance) {
  const landmarks = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  landmarks[9] = { x: 1, y: 0, z: 0 };
  landmarks[4] = { x: 0, y: 0.2, z: 0 };
  landmarks[8] = { x: distance, y: 0.2, z: 0 };
  return landmarks;
}

assert.equal(pinchStrength(landmarksWithPinchDistance(0.18)), 0);
assert.equal(pinchStrength(landmarksWithPinchDistance(1.25)), 1);
assert.equal(pinchStrength(landmarksWithPinchDistance(0.18), true), 1);
assert.equal(pinchStrength([], false), null);
assert.equal(smoothStrength(null, 0.75), 0.75);
assert.equal(smoothStrength(0, 1, 0.25), 0.25);

function foxLandmarks() {
  const points = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.75, z: 0 }));
  points[0] = { x: 0.5, y: 1, z: 0 };
  const setFinger = (indices, x, extended) => {
    const ys = extended ? [0.75, 0.55, 0.35, 0.15] : [0.75, 0.58, 0.66, 0.74];
    indices.forEach((index, offset) => { points[index] = { x, y: ys[offset], z: 0 }; });
  };
  setFinger([5, 6, 7, 8], 0.3, true);
  setFinger([9, 10, 11, 12], 0.45, false);
  setFinger([13, 14, 15, 16], 0.6, false);
  setFinger([17, 18, 19, 20], 0.75, true);
  points[4] = { x: 0.525, y: 0.74, z: 0 };
  return points;
}

assert.ok(foxGestureScore(foxLandmarks()) >= 0.58);

console.log('hand control math: ok');
