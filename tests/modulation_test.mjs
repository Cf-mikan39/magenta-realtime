// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {
  lfoValue,
  mapModulationRange,
  midiCcValue,
} from '../examples/linux_web/modulation.js';

assert.equal(lfoValue('sine', 0), 0);
assert.equal(lfoValue('sine', 0.5), 1);
assert.equal(lfoValue('triangle', 0.5), 1);
assert.equal(lfoValue('square', 0.25), 0);
assert.equal(lfoValue('square', 0.75), 1);
assert.equal(lfoValue('saw', 0.25), 0.25);
assert.equal(mapModulationRange(0.5, 0.2, 0.8), 0.5);
assert.equal(mapModulationRange(0.5, 0.8, 0.2), 0.5);
assert.equal(midiCcValue(0), 0);
assert.equal(midiCcValue(127), 1);
assert.equal(midiCcValue(0, true), 1);

console.log('modulation math: ok');
