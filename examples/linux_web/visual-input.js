import { sharedCamera } from './media-source.js?v=12';

export function normalWeightsValid(prompts, ratio) {
  return ratio === 0 || prompts.some((prompt) => prompt.weight > 0);
}

/** Exclusive image sources, independent of normal prompt editing/modulation. */
export class VisualInput {
  constructor(send, { normalHasWeight = () => true, beforeNormalRatio = () => {} } = {}) {
    this.send = send;
    this.normalHasWeight = normalHasWeight;
    this.beforeNormalRatio = beforeNormalRatio;
    this.elements = Object.fromEntries([
      'mode', 'model', 'ratio', 'timing', 'interval', 'file', 'run', 'disable',
      'video', 'preview', 'tags', 'status', 'share', 'allow-extra-tags',
    ].map((key) => [key, document.querySelector(`#visual-${key}`)]));
    this.generation = 0;
    this.sequence = 0;
    this.enabled = false;
    this.tags = [];
    this.token = null;
    this.ready = false;
    this.acknowledged = -1;
    this.timer = null;
    this.lease = null;
    this.file = null;
    this.previewUrl = null;
    this.upload = null;
    this.captureBusy = false;
    this.captureEpoch = 0;
    this.submitted = false;
    this.awaitingSequence = null;
    this.nextCaptureAt = 0;
    const e = this.elements;
    this.lastRatio = e.ratio.value;
    e.run.addEventListener('click', () => this.start());
    e.disable.addEventListener('click', () => this.disable());
    for (const control of [e.mode, e.model]) {
      control.addEventListener('change', () => { this.disable(); this.render(); });
    }
    e.ratio.addEventListener('change', () => {
      if (this.ratioValue() > 0 && !this.normalHasWeight()) {
        e.ratio.value = this.lastRatio;
        this.status('通常群を使用するには、先に通常テキストの重みを1個以上0%より大きくしてください。');
        this.render();
        return;
      }
      if (this.ratioValue() > 0) this.beforeNormalRatio();
      this.lastRatio = e.ratio.value;
      this.configure();
      this.render();
    });
    e['allow-extra-tags'].addEventListener('change', () => {
      if (!e['allow-extra-tags'].checked && this.tags.length > 5) {
        this.tags = [];
        this.status('5個を超えるタグを適用対象から外しました。静止画は再実行で更新できます。');
      }
      this.configure();
      this.render();
    });
    e.interval.addEventListener('change', () => this.schedule());
    e.timing.addEventListener('change', () => { this.render(); this.schedule(); });
    e.file.addEventListener('change', () => {
      this.disable();
      this.file = e.file.files[0] ?? null;
      if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
      this.previewUrl = this.file ? URL.createObjectURL(this.file) : null;
      e.preview.src = this.previewUrl ?? '';
      this.render();
    });
    this.render();
  }

  config() {
    return { enabled: this.enabled, mode: this.elements.mode.value,
      model: this.elements.model.value, ratio: this.elements.ratio.value,
      allow_extra_tags: Boolean(this.elements['allow-extra-tags'].checked),
      generation: this.generation };
  }

  ratioValue() {
    const [normal, vlm = 1] = this.elements.ratio.value.split(':').map(Number);
    return normal / vlm;
  }

  startPayload() {
    return { visual: this.config(), visual_tags: this.tags };
  }

  status(message) { this.elements.status.textContent = message; }

  render() {
    const still = this.elements.mode.value === 'still';
    this.elements.file.hidden = !still;
    this.elements.preview.hidden = !still || !this.file;
    this.elements.video.hidden = still || !this.lease;
    this.elements.timing.disabled = still;
    this.elements.interval.disabled = still || this.elements.timing.value === 'auto';
    this.elements.tags.textContent = this.tags.join(', ') || 'まだタグはありません';
    const share = this.enabled && this.tags.length
      ? 100 / (this.ratioValue() + 1) : 0;
    this.elements.share.textContent = share
      ? `通常群 ${(100 - share).toFixed(1)}% / VLM群 ${share.toFixed(1)}% · 各タグ ${(share / this.tags.length).toFixed(1)}%`
      : this.ratioValue() === 0
        ? '通常群 0% / VLM群 0% · VLM未適用（通常テキストは使用しません）'
        : '通常群 100% / VLM群 0%';
  }

  configure() {
    this.send({ type: 'visual_config', ...this.config() });
  }

  releaseSource() {
    this.captureEpoch += 1;
    clearTimeout(this.timer);
    this.timer = null;
    this.lease?.release();
    this.lease = null;
    this.elements.video.pause();
    this.elements.video.srcObject = null;
  }

  disable() {
    this.generation += 1;
    this.sequence = 0;
    this.enabled = false;
    this.tags = [];
    this.submitted = false;
    this.awaitingSequence = null;
    this.nextCaptureAt = 0;
    this.upload?.abort();
    this.releaseSource();
    this.configure();
    this.status('画像入力は無効です。通常プロンプトのみ使用します。');
    this.render();
  }

  async start() {
    this.disable();
    this.generation += 1;
    const generation = this.generation;
    const mode = this.elements.mode.value;
    try {
      if (mode === 'still') {
        if (!this.file) throw new Error('画像ファイルを選択してください。');
        if (this.file.size > 8 * 1024 * 1024) throw new Error('画像は8 MiB以下にしてください。');
      } else {
        if (!navigator.mediaDevices) throw new Error('カメラ・画面共有はlocalhostまたはHTTPSで開いてください。');
        const lease = mode === 'camera' ? await sharedCamera.acquire()
          : await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false })
            .then((stream) => ({ stream, release: () => stream.getTracks().forEach((t) => t.stop()) }));
        if (generation !== this.generation) { lease.release(); return; }
        this.lease = lease;
        this.elements.video.srcObject = lease.stream;
        await this.elements.video.play();
        if (generation !== this.generation) return;
        lease.stream.getVideoTracks()[0].addEventListener('ended', () => {
          if (generation === this.generation) this.disable();
        }, { once: true });
      }
      this.enabled = true;
      this.configure();
      this.status(this.ready ? '画像入力を準備しています…' : '入力を準備しました。再生開始で推論します。');
      this.render();
      this.schedule();
    } catch (error) {
      if (generation !== this.generation) return;
      this.disable();
      this.status(error.message);
    }
  }

  connected(token) {
    this.token = token;
    this.ready = false;
    this.acknowledged = -1;
    this.submitted = this.elements.mode.value === 'still' && this.tags.length > 0;
    this.awaitingSequence = null;
    this.nextCaptureAt = 0;
  }

  streamStarted() {
    this.ready = true;
    this.schedule();
  }

  disconnected() {
    this.ready = false;
    this.token = null;
    this.acknowledged = -1;
    this.awaitingSequence = null;
    this.upload?.abort();
    this.releaseSource();
    // A static result survives Stop/Start without another VLM inference.
    if (this.elements.mode.value !== 'still') {
      this.enabled = false;
      this.tags = [];
      this.generation += 1;
    }
    this.render();
  }

  handle(message) {
    if (!message.type.startsWith('visual_')) return false;
    if (message.generation !== this.generation) return true;
    if (message.type === 'visual_state') {
      this.acknowledged = message.generation;
      if (message.tags_cleared) {
        this.tags = [];
        if (this.elements.mode.value === 'still') this.submitted = true;
        this.status('現在の設定ではタグを使用しません。次の推論または静止画の再実行で更新します。');
      }
      this.render();
      this.schedule();
    } else if (message.type === 'visual_processing') {
      this.status('VLMで推論しています。音楽は継続します…');
    } else if (message.type === 'visual_applied') {
      this.completed(message);
      this.tags = message.tags;
      this.status(`適用済み · ${message.elapsed_ms.toFixed(0)} ms`);
      this.render();
      if (this.elements.timing.value === 'auto') this.schedule();
    } else if (message.type === 'visual_error') {
      this.completed(message);
      // Avoid a tight retry loop when the backend or image is unavailable.
      this.nextCaptureAt = Date.now() + 1000;
      this.status(`画像入力: ${message.message}（最後に適用したスタイルを保持）`);
      if (this.elements.timing.value === 'auto') this.schedule();
    }
    return true;
  }

  completed(message) {
    if (this.awaitingSequence !== null && Number.isInteger(message.sequence)
        && message.sequence >= this.awaitingSequence) {
      this.awaitingSequence = null;
    }
  }

  schedule() {
    clearTimeout(this.timer);
    if (!this.ready || !this.enabled || this.acknowledged !== this.generation) return;
    if (this.elements.mode.value === 'still') {
      if (!this.tags.length && !this.submitted) this.capture();
    } else if (this.lease) {
      if (this.captureBusy) return;
      const auto = this.elements.timing.value === 'auto';
      if (auto && this.awaitingSequence !== null) return;
      const interval = !auto && this.submitted
        ? Math.max(0.5, Number(this.elements.interval.value) || 3) * 1000 : 0;
      this.timer = setTimeout(() => this.capture(), Math.max(interval, this.nextCaptureAt - Date.now()));
    }
  }

  async capture() {
    if (this.captureBusy || !this.ready || !this.enabled) return;
    this.captureBusy = true;
    const generation = this.generation;
    const epoch = this.captureEpoch;
    const token = this.token;
    const still = this.elements.mode.value === 'still';
    this.nextCaptureAt = 0;
    if (!still) this.submitted = true;
    try {
      let blob = this.file;
      if (!still) {
        const video = this.elements.video;
        if (!video.videoWidth || video.readyState < 2) {
          this.nextCaptureAt = Date.now() + 500;
          return;
        }
        const scale = Math.min(1, 1024 / Math.max(video.videoWidth, video.videoHeight));
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85));
      }
      if (!blob || generation !== this.generation || epoch !== this.captureEpoch || token !== this.token) return;
      this.upload = new AbortController();
      this.submitted = true;
      const sequence = ++this.sequence;
      this.awaitingSequence = sequence;
      const response = await fetch(`/api/visual/frame/${encodeURIComponent(token)}?generation=${generation}&sequence=${sequence}`, {
        method: 'POST', body: blob, signal: this.upload.signal,
        headers: { 'Content-Type': blob.type || 'application/octet-stream' },
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '画像を送信できませんでした。');
      }
    } catch (error) {
      if (error.name !== 'AbortError' && generation === this.generation && token === this.token) {
        this.awaitingSequence = null;
        this.nextCaptureAt = Date.now() + 1000;
        this.status(error.message);
      }
    } finally {
      this.captureBusy = false;
      this.upload = null;
      this.schedule();
    }
  }
}
