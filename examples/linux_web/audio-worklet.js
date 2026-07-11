/** Consume 48 kHz interleaved stereo float32 frames without main-thread audio. */
class Mrt2PcmPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions ?? {};
    this.frameSamples = config.frameSamples ?? 1_920;
    this.targetFrames = config.targetFrames ?? 3;
    this.maxTargetFrames = config.maxTargetFrames ?? 6;
    this.targetSamples = this.frameSamples * this.targetFrames;
    this.chunks = [];
    this.chunkOffset = 0;
    this.queuedSamples = 0;
    this.playing = false;
    this.underruns = 0;
    this.renderQuanta = 0;

    this.port.onmessage = ({ data }) => {
      if (data.type === 'audio') {
        const chunk = new Float32Array(data.buffer);
        if (chunk.length % 2 !== 0) return;
        this.chunks.push(chunk);
        this.queuedSamples += chunk.length / 2;
      } else if (data.type === 'configure') {
        this.frameSamples = data.frameSamples;
        this.targetFrames = data.targetFrames;
        this.maxTargetFrames = data.maxTargetFrames ?? this.maxTargetFrames;
        this.targetSamples = this.frameSamples * this.targetFrames;
      } else if (data.type === 'reset') {
        this.reset();
      }
    };
  }

  reset() {
    this.chunks = [];
    this.chunkOffset = 0;
    this.queuedSamples = 0;
    this.playing = false;
  }

  copyToOutput(left, right) {
    let outputOffset = 0;
    while (outputOffset < left.length && this.chunks.length > 0) {
      const chunk = this.chunks[0];
      const available = chunk.length / 2 - this.chunkOffset;
      const count = Math.min(left.length - outputOffset, available);
      for (let index = 0; index < count; index += 1) {
        const source = 2 * (this.chunkOffset + index);
        left[outputOffset + index] = chunk[source];
        right[outputOffset + index] = chunk[source + 1];
      }
      outputOffset += count;
      this.chunkOffset += count;
      this.queuedSamples -= count;
      if (this.chunkOffset === chunk.length / 2) {
        this.chunks.shift();
        this.chunkOffset = 0;
      }
    }
    return outputOffset;
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const left = output[0];
    const right = output[1] ?? output[0];
    left.fill(0);
    right.fill(0);

    if (!this.playing && this.queuedSamples >= this.targetSamples) {
      this.playing = true;
    }
    if (this.playing) {
      const copied = this.copyToOutput(left, right);
      if (copied < left.length) {
        this.underruns += 1;
        this.playing = false;
        this.targetFrames = Math.min(
          this.maxTargetFrames,
          this.targetFrames + 1,
        );
        this.targetSamples = this.frameSamples * this.targetFrames;
      }
    }

    this.renderQuanta += 1;
    if (this.renderQuanta % 20 === 0) {
      this.port.postMessage({
        type: 'metrics',
        playing: this.playing,
        bufferMs: (1_000 * this.queuedSamples) / sampleRate,
        underruns: this.underruns,
        targetFrames: this.targetFrames,
      });
    }
    return true;
  }
}

registerProcessor('mrt2-pcm-player', Mrt2PcmPlayer);
