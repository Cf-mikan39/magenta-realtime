// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

"use strict";

export function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value)));
}

export function lfoValue(waveform, phase) {
  const p = ((Number(phase) % 1) + 1) % 1;
  switch (waveform) {
    case "sine":
      return (1 - Math.cos(2 * Math.PI * p)) / 2;
    case "triangle":
      return 1 - Math.abs(2 * p - 1);
    case "square":
      return p < 0.5 ? 0 : 1;
    case "saw":
      return p;
    default:
      throw new Error(`Unsupported LFO waveform: ${waveform}`);
  }
}

export function mapModulationRange(unitValue, minValue, maxValue) {
  const low = clamp01(Math.min(minValue, maxValue));
  const high = clamp01(Math.max(minValue, maxValue));
  return low + clamp01(unitValue) * (high - low);
}

export function midiCcValue(value, invert = false) {
  const unit = clamp01(Number(value) / 127);
  return invert ? 1 - unit : unit;
}
