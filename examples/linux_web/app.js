const elements = {
  prompt: document.querySelector('#prompt'),
  start: document.querySelector('#start'),
  stop: document.querySelector('#stop'),
  apply: document.querySelector('#apply'),
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

let socket = null;
let audioContext = null;
let playerNode = null;
let serverUnderruns = 0;
let browserUnderruns = 0;
let requestId = 0;
let stopping = false;

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

function setControls(isRunning) {
  elements.start.disabled = isRunning;
  elements.stop.disabled = !isRunning;
  elements.apply.disabled = !isRunning;
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
  await audioContext.audioWorklet.addModule('/static/audio-worklet.js');
  playerNode = new AudioWorkletNode(audioContext, 'mrt2-pcm-player', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: {
      frameSamples: 1_920,
      targetFrames: 3,
    },
  });
  playerNode.port.onmessage = ({ data }) => {
    if (data.type !== 'metrics') return;
    elements.browserBuffer.textContent = `${data.bufferMs.toFixed(1)} ms`;
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
      elements.model.textContent = message.model;
      elements.audioFormat.textContent =
        `${message.sample_rate / 1_000} kHz · stereo · ${message.pcm_format}`;
      socket.send(
        JSON.stringify({ type: 'start', prompt: elements.prompt.value }),
      );
      setStatus('初期プロンプトを送信しました。', 'working');
      break;
    case 'status':
      setStatus(message.message, 'working');
      break;
    case 'stream_started':
      playerNode.port.postMessage({
        type: 'configure',
        frameSamples: 1_920,
        targetFrames: message.browser_buffer_frames,
      });
      elements.revision.textContent = message.prompt_revision;
      setStatus('ブラウザの音声バッファを準備しています。', 'working');
      break;
    case 'prompt_encoding':
      elements.apply.disabled = true;
      setStatus(`「${message.prompt}」をエンコード中…`, 'live');
      break;
    case 'prompt_applied':
      elements.apply.disabled = false;
      elements.revision.textContent = message.prompt_revision;
      setStatus(
        `プロンプトを適用しました (${message.encoding_ms.toFixed(1)} ms)。`,
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
      elements.apply.disabled = false;
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
  const prompt = elements.prompt.value.trim();
  if (!prompt) {
    setStatus('プロンプトを入力してください。', 'error');
    elements.prompt.focus();
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
        handleControlMessage(JSON.parse(data));
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
    setControls(false);
    setStatus(error.message, 'error');
  }
}

function applyPrompt() {
  const prompt = elements.prompt.value.trim();
  if (!prompt || !socket || socket.readyState !== WebSocket.OPEN) return;
  requestId += 1;
  socket.send(
    JSON.stringify({ type: 'set_prompt', prompt, request_id: requestId }),
  );
}

function stop() {
  stopping = true;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'stop' }));
    socket.close(1000, 'user stopped playback');
  } else {
    destroyAudioPlayer();
    setControls(false);
    setStatus('停止しました。', 'idle');
    stopping = false;
  }
}

elements.start.addEventListener('click', start);
elements.stop.addEventListener('click', stop);
elements.apply.addEventListener('click', applyPrompt);
elements.prompt.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !elements.apply.disabled) applyPrompt();
});
