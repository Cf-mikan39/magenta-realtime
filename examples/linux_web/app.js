import { HandPromptController } from './hand-control.js?v=7';
import {
  lfoValue,
  mapModulationRange,
  midiCcValue,
} from './modulation.js?v=7';

const COLORS = ['#9b8cff', '#4ed6b2', '#ffb95e', '#ff7891', '#64b5ff', '#d98cff'];
const MAX_PROMPTS = 6;
const WEIGHT_SEND_INTERVAL_MS = 40;
const MODULATION_INTERVAL_MS = 40;
const GESTURE_CONFIRM_FRAMES = 2;
const GESTURE_HOLD_MS = 320;
const GESTURES = [
  { name: 'victory', label: 'ピース' },
  { name: 'open_palm', label: '手のひら' },
  { name: 'fox', label: 'キツネ' },
  { name: 'closed_fist', label: '握りこぶし' },
];
const KEY_TO_SEMITONE = {
  a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7,
  y: 8, h: 9, u: 10, j: 11, k: 12, o: 13, l: 14, p: 15, ';': 16,
};

const elements = {
  start: document.querySelector('#start'),
  stop: document.querySelector('#stop'),
  apply: document.querySelector('#apply'),
  addPrompt: document.querySelector('#add-prompt'),
  listMode: document.querySelector('#list-mode'),
  surfaceMode: document.querySelector('#surface-mode'),
  promptList: document.querySelector('#prompt-list'),
  promptHint: document.querySelector('#prompt-hint'),
  surfaceWrap: document.querySelector('#surface-wrap'),
  surface: document.querySelector('#prompt-surface'),
  surfaceNodes: document.querySelector('#surface-nodes'),
  listener: document.querySelector('#listener'),
  pill: document.querySelector('#connection-pill'),
  status: document.querySelector('#status'),
  model: document.querySelector('#model'),
  revision: document.querySelector('#revision'),
  generationMs: document.querySelector('#generation-ms'),
  serverBuffer: document.querySelector('#server-buffer'),
  browserBuffer: document.querySelector('#browser-buffer'),
  underruns: document.querySelector('#underruns'),
  deadlineMisses: document.querySelector('#deadline-misses'),
  audioFormat: document.querySelector('#audio-format'),
  midiEnable: document.querySelector('#midi-enable'),
  midiInput: document.querySelector('#midi-input'),
  computerKeyboard: document.querySelector('#computer-keyboard'),
  autoStrum: document.querySelector('#auto-strum'),
  midiSolo: document.querySelector('#midi-solo'),
  noDrums: document.querySelector('#no-drums'),
  lfoEnable: document.querySelector('#lfo-enable'),
  lfoTarget: document.querySelector('#lfo-target'),
  lfoWaveform: document.querySelector('#lfo-waveform'),
  lfoRate: document.querySelector('#lfo-rate'),
  lfoRateValue: document.querySelector('#lfo-rate-value'),
  lfoMin: document.querySelector('#lfo-min'),
  lfoMinValue: document.querySelector('#lfo-min-value'),
  lfoMax: document.querySelector('#lfo-max'),
  lfoMaxValue: document.querySelector('#lfo-max-value'),
  lfoCurrent: document.querySelector('#lfo-current'),
  ccEnable: document.querySelector('#cc-enable'),
  ccTarget: document.querySelector('#cc-target'),
  ccNumber: document.querySelector('#cc-number'),
  ccLearn: document.querySelector('#cc-learn'),
  ccInvert: document.querySelector('#cc-invert'),
  ccCurrent: document.querySelector('#cc-current'),
  ccMeterFill: document.querySelector('#cc-meter-fill'),
  ccHint: document.querySelector('#cc-hint'),
  midiPanic: document.querySelector('#midi-panic'),
  midiLed: document.querySelector('#midi-led'),
  midiStatus: document.querySelector('#midi-status'),
  midiNotes: document.querySelector('#midi-notes'),
  octaveDown: document.querySelector('#octave-down'),
  octaveUp: document.querySelector('#octave-up'),
  octaveLabel: document.querySelector('#octave-label'),
  handEnable: document.querySelector('#hand-enable'),
  handPrompt: document.querySelector('#hand-prompt'),
  handInvert: document.querySelector('#hand-invert'),
  handPinchMode: document.querySelector('#hand-pinch-mode'),
  handGestureMode: document.querySelector('#hand-gesture-mode'),
  handPinchControls: document.querySelector('#hand-pinch-controls'),
  handGestureMap: document.querySelector('#hand-gesture-map'),
  gestureVictory: document.querySelector('#gesture-victory'),
  gestureOpenPalm: document.querySelector('#gesture-open-palm'),
  gestureFox: document.querySelector('#gesture-fox'),
  gestureClosedFist: document.querySelector('#gesture-closed-fist'),
  handLed: document.querySelector('#hand-led'),
  handStatus: document.querySelector('#hand-status'),
  handStrength: document.querySelector('#hand-strength'),
  handMeterFill: document.querySelector('#hand-meter-fill'),
  handMeterLabel: document.querySelector('#hand-meter-label'),
  handControlHint: document.querySelector('#hand-control-hint'),
  handVideo: document.querySelector('#hand-video'),
  handCanvas: document.querySelector('#hand-canvas'),
};

let prompts = [
  { id: 0, text: 'disco funk', weight: 1, x: 0.2, y: 0.25 },
  { id: 1, text: 'ambient synth pads', weight: 0, x: 0.8, y: 0.75 },
];
let nextPromptId = 2;
let listenerPosition = { x: 0.5, y: 0.5 };
let mixMode = 'list';
let bankDirty = false;
let socket = null;
let audioContext = null;
let playerNode = null;
let serverUnderruns = 0;
let browserUnderruns = 0;
let requestId = 0;
let stopping = false;
let weightTimer = null;
let surfaceDrag = null;
let midiAccess = null;
let selectedMidiInput = null;
let sustainDown = false;
let keyboardBaseNote = 48;
const activeMidiNotes = new Set();
const heldMidiNotes = new Set();
const pressedComputerKeys = new Map();
let handActive = false;
let handBaseline = new Map();
let handControlMode = 'pinch';
const gestureStates = new Map();
let lfoTimer = null;
let lfoStartedAt = performance.now();
let ccLearning = false;
const gestureSelects = {
  victory: elements.gestureVictory,
  open_palm: elements.gestureOpenPalm,
  fox: elements.gestureFox,
  closed_fist: elements.gestureClosedFist,
};
const handController = new HandPromptController({
  video: elements.handVideo,
  canvas: elements.handCanvas,
  getInvert: () => elements.handInvert.checked,
  onFrame: applyHandFrame,
  onStatus: updateHandStatus,
});

function setStatus(message, state = 'working') {
  elements.status.textContent = message;
  elements.pill.className = `pill ${state}`;
  const labels = {
    idle: '停止中',
    working: '準備中',
    live: 'LIVE',
    error: 'エラー',
  };
  elements.pill.textContent = labels[state] ?? state;
}

function isRunning() {
  return socket !== null && socket.readyState <= WebSocket.OPEN;
}

function setControls(running) {
  elements.start.disabled = running;
  elements.stop.disabled = !running;
  elements.apply.disabled = !running || !bankDirty;
  elements.addPrompt.disabled = prompts.length >= MAX_PROMPTS;
}

function markBankDirty() {
  bankDirty = true;
  setControls(isRunning());
  elements.promptHint.textContent = isRunning()
    ? 'テキストまたは構成が変わりました。「テキスト変更をエンコード」を押してください。'
    : '再生開始時にすべてのプロンプトをエンコードします。';
}

function normalizedWeights() {
  const total = prompts.reduce((sum, prompt) => sum + prompt.weight, 0);
  if (total <= 0) return prompts.map(() => 0);
  return prompts.map((prompt) => prompt.weight / total);
}

function promptPayload() {
  return prompts.map(({ id, text, weight }) => ({
    id,
    text: text.trim(),
    weight,
  }));
}

function weightPayload() {
  return prompts.map(({ id, weight }) => ({ id, weight }));
}

function validatePrompts() {
  if (prompts.some((prompt) => !prompt.text.trim())) {
    throw new Error('すべてのプロンプトにテキストを入力してください。');
  }
  if (prompts.every((prompt) => prompt.weight <= 0)) {
    throw new Error('少なくとも1個の重みを0より大きくしてください。');
  }
}

function renderPrompts() {
  elements.promptList.replaceChildren();
  const normalized = normalizedWeights();

  prompts.forEach((prompt, index) => {
    const row = document.createElement('div');
    row.className = 'prompt-item';
    row.dataset.promptId = prompt.id;

    const color = document.createElement('span');
    color.className = 'prompt-color';
    color.style.background = COLORS[index % COLORS.length];

    const input = document.createElement('input');
    input.className = 'prompt-text';
    input.value = prompt.text;
    input.maxLength = 500;
    input.autocomplete = 'off';
    input.setAttribute('aria-label', `prompt ${index + 1}`);
    input.addEventListener('input', () => {
      prompt.text = input.value;
      markBankDirty();
      renderHandPromptOptions();
      updateSurface();
    });

    const weightControl = document.createElement('div');
    weightControl.className = 'weight-control';
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '1';
    slider.step = '0.01';
    slider.value = String(prompt.weight);
    slider.className = 'weight-slider';
    slider.style.setProperty('--prompt-color', COLORS[index % COLORS.length]);
    slider.addEventListener('input', () => {
      prompt.weight = Number(slider.value);
      mixMode = 'list';
      setMixMode('list');
      updateWeightDisplays();
      scheduleWeightUpdate();
    });
    const weightLabel = document.createElement('output');
    weightLabel.className = 'weight-value';
    weightLabel.dataset.weightFor = prompt.id;
    weightLabel.textContent = `${Math.round(normalized[index] * 100)}%`;
    weightControl.append(slider, weightLabel);

    const remove = document.createElement('button');
    remove.className = 'remove-prompt';
    remove.textContent = '×';
    remove.title = 'プロンプトを削除';
    remove.disabled = prompts.length === 1;
    remove.addEventListener('click', () => {
      prompts = prompts.filter((candidate) => candidate.id !== prompt.id);
      markBankDirty();
      renderPrompts();
      updateSurface();
    });

    row.append(color, input, weightControl, remove);
    elements.promptList.append(row);
  });

  elements.promptList.classList.toggle('surface-active', mixMode === 'surface');
  setControls(isRunning());
  renderHandPromptOptions();
  updateSurface();
}

function updateWeightDisplays() {
  const normalized = normalizedWeights();
  prompts.forEach((prompt, index) => {
    const row = elements.promptList.querySelector(`[data-prompt-id="${prompt.id}"]`);
    if (row) {
      const slider = row.querySelector('.weight-slider');
      const output = row.querySelector('.weight-value');
      if (slider && document.activeElement !== slider) slider.value = prompt.weight;
      if (output) output.textContent = `${Math.round(normalized[index] * 100)}%`;
    }
  });
  updateSurface();
}

function updateSurface() {
  elements.surfaceNodes.replaceChildren();
  const normalized = normalizedWeights();
  prompts.forEach((prompt, index) => {
    const node = document.createElement('button');
    node.className = 'surface-node';
    node.dataset.promptId = prompt.id;
    node.style.left = `${prompt.x * 100}%`;
    node.style.top = `${prompt.y * 100}%`;
    node.style.setProperty('--node-color', COLORS[index % COLORS.length]);
    node.style.setProperty('--node-weight', normalized[index]);
    node.title = prompt.text || `Prompt ${index + 1}`;
    node.innerHTML = `
      <span class="node-dot"></span>
      <span class="node-caption">
        <span>${escapeHtml(prompt.text || `Prompt ${index + 1}`)}</span>
        <b>${Math.round(normalized[index] * 100)}%</b>
      </span>`;
    node.addEventListener('pointerdown', (event) => {
      beginSurfaceDrag(event, 'prompt', prompt.id);
    });
    elements.surfaceNodes.append(node);
  });
  elements.listener.style.left = `${listenerPosition.x * 100}%`;
  elements.listener.style.top = `${listenerPosition.y * 100}%`;
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function calculateSurfaceWeights() {
  const rect = elements.surface.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const distancesSquared = prompts.map((prompt) => {
    const dx = (prompt.x - listenerPosition.x) * width;
    const dy = (prompt.y - listenerPosition.y) * height;
    return dx ** 2 + dy ** 2;
  });
  const exactIndex = distancesSquared.findIndex((distance) => distance < 1);
  if (exactIndex >= 0) {
    return prompts.map((_, index) => (index === exactIndex ? 1 : 0));
  }
  const raw = distancesSquared.map((distance) => 1 / distance);
  const total = raw.reduce((sum, weight) => sum + weight, 0);
  return raw.map((weight) => weight / total);
}

function beginSurfaceDrag(event, type, promptId = null) {
  event.preventDefault();
  surfaceDrag = { type, promptId, pointerId: event.pointerId };
  elements.surface.setPointerCapture(event.pointerId);
  updateSurfaceDrag(event);
}

function updateSurfaceDrag(event) {
  if (!surfaceDrag || event.pointerId !== surfaceDrag.pointerId) return;
  const rect = elements.surface.getBoundingClientRect();
  const x = Math.max(0.04, Math.min(0.96, (event.clientX - rect.left) / rect.width));
  const rawY = (event.clientY - rect.top) / rect.height;
  if (surfaceDrag.type === 'listener') {
    const y = Math.max(0.07, Math.min(0.93, rawY));
    listenerPosition = { x, y };
  } else {
    const prompt = prompts.find((candidate) => candidate.id === surfaceDrag.promptId);
    if (prompt) {
      const y = Math.max(0.07, Math.min(0.82, rawY));
      prompt.x = x;
      prompt.y = y;
    }
  }
  const weights = calculateSurfaceWeights();
  prompts.forEach((prompt, index) => { prompt.weight = weights[index]; });
  updateWeightDisplays();
  scheduleWeightUpdate();
}

function endSurfaceDrag(event) {
  if (!surfaceDrag || event.pointerId !== surfaceDrag.pointerId) return;
  if (elements.surface.hasPointerCapture(event.pointerId)) {
    elements.surface.releasePointerCapture(event.pointerId);
  }
  surfaceDrag = null;
}

function setMixMode(mode) {
  mixMode = mode;
  const surfaceActive = mode === 'surface';
  elements.surfaceWrap.hidden = !surfaceActive;
  elements.listMode.classList.toggle('active', !surfaceActive);
  elements.surfaceMode.classList.toggle('active', surfaceActive);
  elements.promptList.classList.toggle('surface-active', surfaceActive);
  if (surfaceActive) {
    const weights = calculateSurfaceWeights();
    prompts.forEach((prompt, index) => { prompt.weight = weights[index]; });
    updateWeightDisplays();
    scheduleWeightUpdate();
  }
}

function scheduleWeightUpdate() {
  if (!isRunning() || bankDirty) return;
  if (prompts.every((prompt) => prompt.weight <= 0)) {
    setStatus('少なくとも1個の重みを0より大きくしてください。', 'error');
    return;
  }
  if (weightTimer !== null) return;
  weightTimer = window.setTimeout(() => {
    weightTimer = null;
    if (!socket || socket.readyState !== WebSocket.OPEN || bankDirty) return;
    socket.send(JSON.stringify({ type: 'set_weights', weights: weightPayload() }));
  }, WEIGHT_SEND_INTERVAL_MS);
}

function applyModulatedWeight(promptId, value) {
  const target = prompts.find((prompt) => prompt.id === promptId);
  if (!target) return;
  target.weight = Math.max(0, Math.min(1, value));
  if (prompts.every((prompt) => prompt.weight <= 0)) {
    const fallback = prompts.find((prompt) => prompt.id !== promptId);
    if (fallback) fallback.weight = 1;
    else target.weight = 0.001;
  }
  updateWeightDisplays();
  scheduleWeightUpdate();
}

function syncLfoControls(changed = null) {
  let minValue = Number(elements.lfoMin.value);
  let maxValue = Number(elements.lfoMax.value);
  if (minValue > maxValue) {
    if (changed === 'min') {
      maxValue = minValue;
      elements.lfoMax.value = String(maxValue);
    } else {
      minValue = maxValue;
      elements.lfoMin.value = String(minValue);
    }
  }
  elements.lfoRateValue.textContent = `${Number(elements.lfoRate.value).toFixed(2)} Hz`;
  elements.lfoMinValue.textContent = `${Math.round(minValue * 100)}%`;
  elements.lfoMaxValue.textContent = `${Math.round(maxValue * 100)}%`;
}

function runLfoFrame() {
  const rate = Number(elements.lfoRate.value);
  const phase = ((performance.now() - lfoStartedAt) / 1000 * rate) % 1;
  const unit = lfoValue(elements.lfoWaveform.value, phase);
  const value = mapModulationRange(
    unit,
    Number(elements.lfoMin.value),
    Number(elements.lfoMax.value),
  );
  elements.lfoCurrent.textContent = `${Math.round(value * 100)}%`;
  applyModulatedWeight(Number(elements.lfoTarget.value), value);
}

function setLfoEnabled(enabled) {
  elements.lfoEnable.checked = enabled;
  if (lfoTimer !== null) {
    window.clearInterval(lfoTimer);
    lfoTimer = null;
  }
  if (!enabled) {
    elements.lfoCurrent.textContent = '—';
    return;
  }
  lfoStartedAt = performance.now();
  runLfoFrame();
  lfoTimer = window.setInterval(runLfoFrame, MODULATION_INTERVAL_MS);
}

function configuredCcNumber() {
  const value = Math.max(
    0,
    Math.min(119, Math.round(Number(elements.ccNumber.value) || 0)),
  );
  elements.ccNumber.value = String(value);
  return value;
}

function updateCcHint(message = null) {
  if (message) {
    elements.ccHint.textContent = message;
    return;
  }
  const number = configuredCcNumber();
  elements.ccHint.textContent = elements.ccEnable.checked
    ? `選択中のMIDI入力からCC ${number}を受信するとプロンプトを変調します。`
    : `MIDI CC変調は停止中です（割り当て: CC ${number}）。`;
}

function beginCcLearn() {
  ccLearning = !ccLearning;
  elements.ccLearn.textContent = ccLearning ? 'Move a knob…' : 'Learn';
  updateCcHint(
    ccLearning
      ? '割り当てたいMIDIコントローラーを動かしてください。'
      : null,
  );
}

function handleMidiCc(controller, rawValue) {
  if (ccLearning && controller <= 119) {
    elements.ccNumber.value = String(controller);
    ccLearning = false;
    elements.ccLearn.textContent = 'Learn';
    updateCcHint(`CC ${controller}を割り当てました。`);
  }
  if (!elements.ccEnable.checked || controller !== configuredCcNumber()) return;
  const value = midiCcValue(rawValue, elements.ccInvert.checked);
  elements.ccCurrent.textContent = `CC${controller} · ${Math.round(value * 100)}%`;
  elements.ccMeterFill.style.width = `${value * 100}%`;
  applyModulatedWeight(Number(elements.ccTarget.value), value);
}

function renderHandPromptOptions() {
  const previousId = Number(elements.handPrompt.value);
  populatePromptSelect(elements.handPrompt, previousId, 0);
  GESTURES.forEach((gesture, index) => {
    const select = gestureSelects[gesture.name];
    const selectedId = select.options.length > 0 ? Number(select.value) : NaN;
    populatePromptSelect(select, selectedId, index % prompts.length);
  });
  const lfoTargetId = elements.lfoTarget.options.length > 0
    ? Number(elements.lfoTarget.value)
    : NaN;
  const ccTargetId = elements.ccTarget.options.length > 0
    ? Number(elements.ccTarget.value)
    : NaN;
  const modulationDefault = Math.min(1, prompts.length - 1);
  populatePromptSelect(elements.lfoTarget, lfoTargetId, modulationDefault);
  populatePromptSelect(elements.ccTarget, ccTargetId, modulationDefault);
  elements.handPrompt.disabled = prompts.length < 2;
  elements.handEnable.disabled = prompts.length < 2;
  if (prompts.length < 2 && handActive) {
    handController.stop();
    handActive = false;
    setHandModeLock(false);
    elements.handEnable.textContent = 'カメラを有効化';
  }
  if (handActive) captureHandBaseline();
}

function populatePromptSelect(select, previousId, defaultIndex) {
  select.replaceChildren();
  prompts.forEach((prompt, index) => {
    const option = document.createElement('option');
    option.value = String(prompt.id);
    option.textContent = prompt.text.trim() || `Prompt ${index + 1}`;
    select.append(option);
  });
  const selected = prompts.some((prompt) => prompt.id === previousId)
    ? previousId
    : prompts[Math.min(defaultIndex, prompts.length - 1)].id;
  select.value = String(selected);
}

function captureHandBaseline() {
  const targetId = Number(elements.handPrompt.value);
  const otherPrompts = prompts.filter((prompt) => prompt.id !== targetId);
  const total = otherPrompts.reduce((sum, prompt) => sum + prompt.weight, 0);
  handBaseline = new Map();
  for (const prompt of otherPrompts) {
    const share = total > 0
      ? prompt.weight / total
      : 1 / Math.max(1, otherPrompts.length);
    handBaseline.set(prompt.id, share);
  }
}

function applyHandFrame({ hands }) {
  elements.handLed.classList.toggle('tracking', hands.length > 0);
  if (!handActive) return;
  if (handControlMode === 'gesture') {
    applyGestureFrame(hands);
    return;
  }
  const hand = hands[0];
  const strength = hand?.pinchStrength;
  if (strength === null || !Number.isFinite(strength)) return;
  const value = Math.max(0, Math.min(1, strength));
  const targetId = Number(elements.handPrompt.value);
  for (const prompt of prompts) {
    prompt.weight = prompt.id === targetId
      ? value
      : (handBaseline.get(prompt.id) ?? 0) * (1 - value);
  }
  elements.handStrength.textContent = `${Math.round(value * 100)}%`;
  elements.handMeterFill.style.width = `${value * 100}%`;
  elements.handStatus.textContent = `${hand.handedness}: Pinch ${Math.round(value * 100)}%`;
  updateWeightDisplays();
  scheduleWeightUpdate();
}

function applyGestureFrame(hands, now = performance.now()) {
  const seenHands = new Set();
  for (const hand of hands) {
    const key = hand.handedness;
    seenHands.add(key);
    const state = gestureStates.get(key) ?? {
      candidate: null,
      candidateFrames: 0,
      active: null,
      confidence: 0,
      lastRecognizedMs: -Infinity,
    };
    const recognized = GESTURES.some(({ name }) => name === hand.gesture)
      && hand.gestureConfidence >= 0.45;
    if (recognized) {
      if (state.candidate === hand.gesture) {
        state.candidateFrames += 1;
      } else {
        state.candidate = hand.gesture;
        state.candidateFrames = 1;
      }
      if (state.candidateFrames >= GESTURE_CONFIRM_FRAMES) {
        const gestureChanged = state.active !== hand.gesture;
        state.active = hand.gesture;
        state.confidence = !gestureChanged && state.confidence > 0
          ? 0.72 * state.confidence + 0.28 * hand.gestureConfidence
          : hand.gestureConfidence;
        state.lastRecognizedMs = now;
      }
    } else {
      state.candidate = null;
      state.candidateFrames = 0;
    }
    gestureStates.set(key, state);
  }

  for (const [key, state] of gestureStates) {
    if (!seenHands.has(key) || now - state.lastRecognizedMs > GESTURE_HOLD_MS) {
      if (now - state.lastRecognizedMs > GESTURE_HOLD_MS) {
        state.active = null;
        state.confidence = 0;
      }
    }
  }

  const activeHands = [...gestureStates.entries()]
    .filter(([, state]) => state.active !== null);
  if (activeHands.length === 0) {
    elements.handStrength.textContent = '—';
    elements.handMeterFill.style.width = '0%';
    return;
  }

  const contributions = new Map();
  for (const [, state] of activeHands) {
    const select = gestureSelects[state.active];
    const promptId = Number(select.value);
    contributions.set(
      promptId,
      Math.max(contributions.get(promptId) ?? 0, state.confidence),
    );
  }
  const total = [...contributions.values()].reduce((sum, value) => sum + value, 0);
  if (total <= 0) return;
  for (const prompt of prompts) {
    prompt.weight = (contributions.get(prompt.id) ?? 0) / total;
  }

  const descriptions = activeHands.map(([hand, state]) => {
    const label = GESTURES.find(({ name }) => name === state.active)?.label;
    return `${hand}: ${label} ${Math.round(state.confidence * 100)}%`;
  });
  elements.handStatus.textContent = descriptions.join(' · ');
  elements.handStrength.textContent = `${activeHands.length} hand`;
  elements.handMeterFill.style.width = `${Math.min(100, activeHands.length * 50)}%`;
  updateWeightDisplays();
  scheduleWeightUpdate();
}

function setHandControlMode(mode) {
  handControlMode = mode;
  const gestureMode = mode === 'gesture';
  elements.handPinchControls.hidden = gestureMode;
  elements.handGestureMap.hidden = !gestureMode;
  elements.handPinchMode.classList.toggle('active', !gestureMode);
  elements.handGestureMode.classList.toggle('active', gestureMode);
  elements.handMeterLabel.textContent = gestureMode
    ? 'Active gestures'
    : 'Prompt strength';
  elements.handControlHint.textContent = gestureMode
    ? 'ジェスチャーを2フレーム連続で認識すると適用し、短い見失いでは直前の状態を保持します。'
    : '親指と人差し指を閉じると0%、広げると100%。選択したプロンプト以外の比率を保ったまま連続制御します。';
  gestureStates.clear();
  elements.handStrength.textContent = '—';
  elements.handMeterFill.style.width = '0%';
  if (!gestureMode) captureHandBaseline();
}

function updateHandStatus(state, message) {
  elements.handStatus.textContent = message;
  elements.handLed.className = `hand-led ${state}`;
  if (state === 'error') {
    handActive = false;
    setHandModeLock(false);
    elements.handEnable.disabled = prompts.length < 2;
    elements.handEnable.textContent = 'カメラを有効化';
  }
}

function setHandModeLock(locked) {
  elements.listMode.disabled = locked;
  elements.surfaceMode.disabled = locked;
}

async function toggleHandControl() {
  if (handActive) {
    handController.stop();
    handActive = false;
    setHandModeLock(false);
    elements.handEnable.textContent = 'カメラを有効化';
    return;
  }
  if (prompts.length < 2) {
    updateHandStatus('error', '手制御には2個以上のプロンプトが必要です。');
    return;
  }

  setMixMode('list');
  captureHandBaseline();
  elements.handEnable.disabled = true;
  try {
    await handController.start();
    handActive = true;
    setHandModeLock(true);
    elements.handEnable.disabled = false;
    elements.handEnable.textContent = 'カメラを停止';
  } catch (error) {
    handController.stop(false);
    handActive = false;
    setHandModeLock(false);
    elements.handEnable.disabled = false;
    elements.handEnable.textContent = 'カメラを有効化';
    updateHandStatus('error', `カメラを開始できません: ${error.message}`);
  }
}

function midiConditioningEnabled() {
  return selectedMidiInput !== null || elements.computerKeyboard.checked;
}

function sendMidiConfig() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: 'midi_config',
    enabled: midiConditioningEnabled(),
    auto_strum: elements.autoStrum.checked,
    unmask_width: elements.midiSolo.checked ? 127 : 4,
  }));
}

function sendDrumConfig() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: 'drum_config',
    no_drums: elements.noDrums.checked,
  }));
}

function sendMidiNote(pitch, on) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ type: 'midi_note', pitch, on }));
}

function midiNoteOn(pitch) {
  if (!Number.isInteger(pitch) || pitch < 0 || pitch > 127) return;
  heldMidiNotes.add(pitch);
  activeMidiNotes.add(pitch);
  sendMidiNote(pitch, true);
  updateMidiUi();
}

function midiNoteOff(pitch) {
  heldMidiNotes.delete(pitch);
  if (sustainDown) return;
  if (activeMidiNotes.delete(pitch)) sendMidiNote(pitch, false);
  updateMidiUi();
}

function setSustain(down) {
  if (sustainDown === down) return;
  sustainDown = down;
  if (!sustainDown) {
    for (const pitch of [...activeMidiNotes]) {
      if (!heldMidiNotes.has(pitch)) {
        activeMidiNotes.delete(pitch);
        sendMidiNote(pitch, false);
      }
    }
  }
  updateMidiUi();
}

function allNotesOff() {
  activeMidiNotes.clear();
  heldMidiNotes.clear();
  pressedComputerKeys.clear();
  sustainDown = false;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'midi_all_notes_off' }));
  }
  updateMidiUi();
}

function midiNoteName(pitch) {
  const names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
  return `${names[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

function updateMidiUi() {
  const notes = [...activeMidiNotes].sort((a, b) => a - b);
  elements.midiLed.classList.toggle('active', notes.length > 0);
  elements.midiNotes.innerHTML = notes.length > 0
    ? notes.map((pitch) => `<span class="midi-note">${midiNoteName(pitch)}<b>${pitch}</b></span>`).join('')
    : '<span>Active notes: —</span>';

  const sources = [];
  if (selectedMidiInput) sources.push(selectedMidiInput.name || 'MIDI device');
  if (elements.computerKeyboard.checked) sources.push('PC keyboard');
  elements.midiStatus.textContent = sources.length > 0
    ? `${sources.join(' + ')}${sustainDown ? ' · Sustain' : ''}`
    : (midiAccess ? '入力を選択してください' : '未接続');
  elements.octaveLabel.textContent = `Keyboard C${Math.floor(keyboardBaseNote / 12) - 1}`;
}

function attachMidiInput(input) {
  if (selectedMidiInput) selectedMidiInput.onmidimessage = null;
  allNotesOff();
  selectedMidiInput = input;
  if (selectedMidiInput) selectedMidiInput.onmidimessage = handleMidiMessage;
  sendMidiConfig();
  updateMidiUi();
}

function renderMidiInputs() {
  const inputs = midiAccess ? [...midiAccess.inputs.values()] : [];
  const connected = inputs.filter((input) => input.state !== 'disconnected');
  const previousId = selectedMidiInput?.id ?? '';
  elements.midiInput.replaceChildren();

  const none = document.createElement('option');
  none.value = '';
  none.textContent = connected.length > 0 ? '入力を選択' : 'MIDIデバイスなし';
  elements.midiInput.append(none);
  for (const input of connected) {
    const option = document.createElement('option');
    option.value = input.id;
    option.textContent = input.name || `MIDI ${input.id}`;
    elements.midiInput.append(option);
  }
  elements.midiInput.disabled = connected.length === 0;

  const next = connected.find((input) => input.id === previousId) ?? connected[0] ?? null;
  elements.midiInput.value = next?.id ?? '';
  if (next !== selectedMidiInput) attachMidiInput(next);
  updateMidiUi();
}

async function enableWebMidi() {
  if (!navigator.requestMIDIAccess) {
    elements.midiStatus.textContent = 'このブラウザはWeb MIDI非対応です';
    setStatus('ChromeまたはEdgeでWeb MIDIを使用してください。', 'error');
    return;
  }
  try {
    midiAccess = await navigator.requestMIDIAccess({ sysex: false });
    midiAccess.onstatechange = renderMidiInputs;
    elements.midiEnable.textContent = 'MIDIを再スキャン';
    renderMidiInputs();
  } catch (error) {
    elements.midiStatus.textContent = 'MIDIアクセス拒否';
    setStatus(`MIDIを開始できません: ${error.message}`, 'error');
  }
}

function handleMidiMessage(event) {
  const [statusByte, data1, data2] = event.data;
  const status = statusByte & 0xf0;
  if (status === 0x90 && data2 > 0) {
    midiNoteOn(data1);
  } else if (status === 0x80 || (status === 0x90 && data2 === 0)) {
    midiNoteOff(data1);
  } else if (status === 0xb0) {
    handleMidiCc(data1, data2);
    if (data1 === 64) {
      setSustain(data2 >= 64);
    } else if (data1 === 120 || data1 === 123) {
      allNotesOff();
    }
  }
}

function computerKeyDown(event) {
  if (!elements.computerKeyboard.checked || event.repeat) return;
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === 'z' || key === 'x') {
    keyboardBaseNote = Math.max(
      24,
      Math.min(84, keyboardBaseNote + (key === 'z' ? -12 : 12)),
    );
    updateMidiUi();
    event.preventDefault();
    return;
  }
  if (!(key in KEY_TO_SEMITONE)) return;
  const pitch = keyboardBaseNote + KEY_TO_SEMITONE[key];
  pressedComputerKeys.set(key, pitch);
  midiNoteOn(pitch);
  event.preventDefault();
}

function computerKeyUp(event) {
  const key = event.key.toLowerCase();
  const pitch = pressedComputerKeys.get(key);
  if (pitch === undefined) return;
  pressedComputerKeys.delete(key);
  midiNoteOff(pitch);
  event.preventDefault();
}

function updateUnderruns() {
  elements.underruns.textContent = `${serverUnderruns} / ${browserUnderruns}`;
}

async function createAudioPlayer() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error('このブラウザはWeb Audio APIに対応していません。');
  }
  audioContext = new AudioContextClass({
    sampleRate: 48_000,
    latencyHint: 'interactive',
  });
  if (audioContext.sampleRate !== 48_000) {
    const actualSampleRate = audioContext.sampleRate;
    await audioContext.close();
    audioContext = null;
    throw new Error(
      `AudioContextが48 kHzではありません (${actualSampleRate} Hz)。`,
    );
  }
  await audioContext.audioWorklet.addModule('/static/audio-worklet.js?v=7');
  playerNode = new AudioWorkletNode(audioContext, 'mrt2-pcm-player', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: {
      frameSamples: 1_920,
      targetFrames: 3,
      maxTargetFrames: 6,
    },
  });
  playerNode.port.onmessage = ({ data }) => {
    if (data.type !== 'metrics') return;
    elements.browserBuffer.textContent =
      `${data.bufferMs.toFixed(1)} ms · target ${data.targetFrames * 40} ms`;
    browserUnderruns = data.underruns;
    updateUnderruns();
    if (data.playing && elements.pill.classList.contains('working')) {
      setStatus('リアルタイム音声を再生しています。', 'live');
    }
  };
  playerNode.connect(audioContext.destination);
  await audioContext.resume();
}

async function destroyAudioPlayer() {
  if (playerNode) {
    playerNode.port.postMessage({ type: 'reset' });
    playerNode.disconnect();
    playerNode = null;
  }
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }
}

function handleControlMessage(message) {
  switch (message.type) {
    case 'hello':
      if (message.protocol_version < 3) {
        throw new Error('サーバーのWebSocketプロトコルが古いバージョンです。');
      }
      elements.model.textContent = message.model;
      elements.audioFormat.textContent =
        `${message.sample_rate / 1_000} kHz · stereo · ${message.pcm_format}`;
      socket.send(JSON.stringify({ type: 'start', prompts: promptPayload() }));
      setStatus(`${prompts.length}個の初期プロンプトを送信しました。`, 'working');
      break;
    case 'status':
      setStatus(message.message, 'working');
      break;
    case 'stream_started':
      playerNode.port.postMessage({
        type: 'configure',
        frameSamples: 1_920,
        targetFrames: message.browser_buffer_frames,
        maxTargetFrames: message.browser_max_buffer_frames,
      });
      bankDirty = false;
      setControls(true);
      sendMidiConfig();
      sendDrumConfig();
      for (const pitch of activeMidiNotes) sendMidiNote(pitch, true);
      elements.revision.textContent = message.prompt_revision;
      elements.promptHint.textContent =
        '重み変更は即時反映されます。テキスト変更のみ再エンコードが必要です。';
      setStatus('ブラウザの音声バッファを準備しています。', 'working');
      break;
    case 'prompt_encoding':
      elements.apply.disabled = true;
      setStatus(
        message.uncached_count > 0
          ? `${message.uncached_count}個の新しいプロンプトをエンコードします。`
          : 'キャッシュ済み埋め込みを再利用しています。',
        'live',
      );
      break;
    case 'prompt_encoding_progress':
      setStatus(
        `「${message.prompt}」をエンコード中 (${message.completed + 1}/${message.total})…`,
        'live',
      );
      break;
    case 'prompt_applied':
      bankDirty = false;
      setControls(true);
      elements.revision.textContent = message.prompt_revision;
      elements.promptHint.textContent =
        '重み変更は即時反映されます。テキスト変更のみ再エンコードが必要です。';
      setStatus(
        `プロンプトバンクを適用しました (${message.encoding_ms.toFixed(1)} ms)。`,
        'live',
      );
      break;
    case 'midi_config_applied':
      updateMidiUi();
      break;
    case 'drum_config_applied':
      elements.noDrums.checked = message.no_drums;
      break;
    case 'metrics':
      elements.generationMs.textContent =
        `${message.generation_ms_latest.toFixed(1)} ms`;
      elements.serverBuffer.textContent =
        `${message.server_buffer_ms.toFixed(1)} ms`;
      elements.deadlineMisses.textContent = message.generation_deadline_misses;
      elements.revision.textContent = message.prompt_revision;
      serverUnderruns = message.server_underrun_frames;
      updateUnderruns();
      break;
    case 'control_error':
      setControls(true);
      setStatus(message.message, 'error');
      break;
    case 'error':
      setStatus(message.message, 'error');
      break;
    default:
      console.debug('Unknown MRT2 message', message);
  }
}

async function start() {
  try {
    validatePrompts();
  } catch (error) {
    setStatus(error.message, 'error');
    return;
  }
  setControls(true);
  setStatus('AudioWorkletを準備しています。', 'working');
  serverUnderruns = 0;
  browserUnderruns = 0;
  updateUnderruns();

  try {
    await createAudioPlayer();
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${location.host}/ws/audio`);
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => setStatus('サーバーに接続しました。', 'working');
    socket.onmessage = ({ data }) => {
      if (typeof data === 'string') {
        try {
          handleControlMessage(JSON.parse(data));
        } catch (error) {
          setStatus(error.message, 'error');
        }
      } else if (data instanceof ArrayBuffer && playerNode) {
        playerNode.port.postMessage({ type: 'audio', buffer: data }, [data]);
      }
    };
    socket.onerror = () => setStatus('WebSocket通信に失敗しました。', 'error');
    socket.onclose = async () => {
      socket = null;
      allNotesOff();
      await destroyAudioPlayer();
      setControls(false);
      if (stopping) {
        setStatus('停止しました。', 'idle');
      } else if (!elements.pill.classList.contains('error')) {
        setStatus('サーバーとの接続が終了しました。', 'idle');
      }
      stopping = false;
    };
  } catch (error) {
    await destroyAudioPlayer();
    socket = null;
    setControls(false);
    setStatus(error.message, 'error');
  }
}

function applyPrompts() {
  try {
    validatePrompts();
  } catch (error) {
    setStatus(error.message, 'error');
    return;
  }
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  requestId += 1;
  elements.apply.disabled = true;
  socket.send(JSON.stringify({
    type: 'set_prompts',
    prompts: promptPayload(),
    request_id: requestId,
  }));
}

function stop() {
  stopping = true;
  allNotesOff();
  if (weightTimer !== null) {
    clearTimeout(weightTimer);
    weightTimer = null;
  }
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'stop' }));
    socket.close(1000, 'user stopped playback');
  } else {
    socket = null;
    destroyAudioPlayer();
    setControls(false);
    setStatus('停止しました。', 'idle');
    stopping = false;
  }
}

elements.start.addEventListener('click', start);
elements.stop.addEventListener('click', stop);
elements.apply.addEventListener('click', applyPrompts);
elements.addPrompt.addEventListener('click', () => {
  if (prompts.length >= MAX_PROMPTS) return;
  const angle = (2 * Math.PI * prompts.length) / MAX_PROMPTS;
  prompts.push({
    id: nextPromptId,
    text: '',
    weight: 0.5,
    x: 0.5 + 0.32 * Math.cos(angle),
    y: 0.5 + 0.32 * Math.sin(angle),
  });
  nextPromptId += 1;
  markBankDirty();
  renderPrompts();
});
elements.listMode.addEventListener('click', () => setMixMode('list'));
elements.surfaceMode.addEventListener('click', () => setMixMode('surface'));
elements.listener.addEventListener('pointerdown', (event) => {
  beginSurfaceDrag(event, 'listener');
});
elements.surface.addEventListener('pointermove', updateSurfaceDrag);
elements.surface.addEventListener('pointerup', endSurfaceDrag);
elements.surface.addEventListener('pointercancel', endSurfaceDrag);
elements.midiEnable.addEventListener('click', enableWebMidi);
elements.midiInput.addEventListener('change', () => {
  const input = midiAccess?.inputs.get(elements.midiInput.value) ?? null;
  attachMidiInput(input);
});
elements.computerKeyboard.addEventListener('change', () => {
  allNotesOff();
  sendMidiConfig();
  updateMidiUi();
});
elements.autoStrum.addEventListener('change', sendMidiConfig);
elements.midiSolo.addEventListener('change', sendMidiConfig);
elements.noDrums.addEventListener('change', sendDrumConfig);
elements.lfoEnable.addEventListener('change', () => {
  setLfoEnabled(elements.lfoEnable.checked);
});
elements.lfoWaveform.addEventListener('change', () => {
  lfoStartedAt = performance.now();
  if (elements.lfoEnable.checked) runLfoFrame();
});
elements.lfoRate.addEventListener('input', () => syncLfoControls());
elements.lfoMin.addEventListener('input', () => syncLfoControls('min'));
elements.lfoMax.addEventListener('input', () => syncLfoControls('max'));
elements.ccEnable.addEventListener('change', () => {
  if (!elements.ccEnable.checked) {
    elements.ccCurrent.textContent = '—';
    elements.ccMeterFill.style.width = '0%';
  }
  updateCcHint();
});
elements.ccNumber.addEventListener('change', () => updateCcHint());
elements.ccLearn.addEventListener('click', beginCcLearn);
elements.midiPanic.addEventListener('click', allNotesOff);
elements.handEnable.addEventListener('click', toggleHandControl);
elements.handPrompt.addEventListener('change', captureHandBaseline);
elements.handPinchMode.addEventListener('click', () => setHandControlMode('pinch'));
elements.handGestureMode.addEventListener('click', () => setHandControlMode('gesture'));
elements.octaveDown.addEventListener('click', () => {
  keyboardBaseNote = Math.max(24, keyboardBaseNote - 12);
  updateMidiUi();
});
elements.octaveUp.addEventListener('click', () => {
  keyboardBaseNote = Math.min(84, keyboardBaseNote + 12);
  updateMidiUi();
});
window.addEventListener('keydown', computerKeyDown);
window.addEventListener('keyup', computerKeyUp);
window.addEventListener('blur', () => {
  for (const pitch of pressedComputerKeys.values()) midiNoteOff(pitch);
  pressedComputerKeys.clear();
});
if ('ResizeObserver' in window) {
  const surfaceResizeObserver = new ResizeObserver(() => {
    if (mixMode !== 'surface' || elements.surfaceWrap.hidden) return;
    const weights = calculateSurfaceWeights();
    prompts.forEach((prompt, index) => { prompt.weight = weights[index]; });
    updateWeightDisplays();
    scheduleWeightUpdate();
  });
  surfaceResizeObserver.observe(elements.surface);
}

renderPrompts();
setMixMode('list');
setHandControlMode('pinch');
setControls(false);
updateMidiUi();
syncLfoControls();
updateCcHint();
