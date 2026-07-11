const COLORS = ['#9b8cff', '#4ed6b2', '#ffb95e', '#ff7891', '#64b5ff', '#d98cff'];
const MAX_PROMPTS = 6;
const WEIGHT_SEND_INTERVAL_MS = 40;

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
    node.innerHTML = `<span>${escapeHtml(prompt.text || `Prompt ${index + 1}`)}</span><b>${Math.round(normalized[index] * 100)}%</b>`;
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
  const distancesSquared = prompts.map((prompt) =>
    (prompt.x - listenerPosition.x) ** 2 +
    (prompt.y - listenerPosition.y) ** 2
  );
  const exactIndex = distancesSquared.findIndex((distance) => distance < 0.00001);
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
  const y = Math.max(0.07, Math.min(0.93, (event.clientY - rect.top) / rect.height));
  if (surfaceDrag.type === 'listener') {
    listenerPosition = { x, y };
  } else {
    const prompt = prompts.find((candidate) => candidate.id === surfaceDrag.promptId);
    if (prompt) {
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
  await audioContext.audioWorklet.addModule('/static/audio-worklet.js?v=2');
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
      if (message.protocol_version < 2) {
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

renderPrompts();
setMixMode('list');
setControls(false);
