/** Browser state tests with fake devices/transport; no camera or server used. */
import assert from 'node:assert/strict';
import { CameraSource } from '../examples/linux_web/media-source.js';
import { VisualInput, normalWeightsValid } from '../examples/linux_web/visual-input.js';

const zeroWeights = [{ weight: 0 }, { weight: 0 }];
assert.equal(normalWeightsValid(zeroWeights, 0), true);
assert.equal(normalWeightsValid(zeroWeights, 2), false);
assert.equal(normalWeightsValid([{ weight: 0 }, { weight: .5 }], 2), true);

let stops = 0;
let acquisitions = 0;
const track = { readyState: 'live', stop: () => { stops += 1; } };
const stream = { getVideoTracks: () => [track], getTracks: () => [track] };
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {
  mediaDevices: { getUserMedia: async () => { acquisitions += 1; return stream; } },
} });
const source = new CameraSource();
const [hand, vlm] = await Promise.all([source.acquire(), source.acquire()]);
assert.equal(acquisitions, 1);
hand.release();
assert.equal(stops, 0);
vlm.release();
vlm.release();
assert.equal(stops, 1);

const controls = new Map();
globalThis.document = { querySelector: (id) => {
  if (!controls.has(id)) controls.set(id, {
    value: '', files: [], hidden: false, textContent: '', srcObject: null,
    listeners: {}, addEventListener(name, fn) { this.listeners[name] = fn; },
    pause() {}, async play() {},
  });
  return controls.get(id);
} };
for (const [name, value] of Object.entries({ mode: 'still', model: 'gemma', ratio: '2:1', interval: '3', timing: 'interval' })) {
  document.querySelector(`#visual-${name}`).value = value;
}
const messages = [];
let normalHasWeight = true;
let normalRatioPrepares = 0;
const visual = new VisualInput((m) => { messages.push(m); return true; }, {
  normalHasWeight: () => normalHasWeight,
  beforeNormalRatio: () => { normalRatioPrepares += 1; },
});
visual.file = new Blob(['fake image'], { type: 'image/jpeg' });
visual.connected('token');
await visual.start();
const generation = visual.generation;
assert.equal(visual.enabled, true);
assert.equal(visual.config().allow_extra_tags, false);
assert.equal(messages.at(-1).model, 'gemma');
assert(messages.at(-1).generation > messages.at(-2).generation);
visual.handle({ type: 'visual_state', generation });
let uploads = 0;
globalThis.fetch = async () => { uploads += 1; return { ok: true }; };
visual.streamStarted();
await new Promise((resolve) => setImmediate(resolve));
assert.equal(uploads, 1);
visual.schedule();
assert.equal(uploads, 1);
visual.handle({ type: 'visual_applied', generation, tags: ['calm', 'piano'], elapsed_ms: 20 });
assert.deepEqual(visual.tags, ['calm', 'piano']);
assert(visual.elements.share.textContent.includes('16.7%'));
visual.handle({ type: 'visual_applied', generation: generation - 1, tags: ['old'], elapsed_ms: 20 });
assert.deepEqual(visual.tags, ['calm', 'piano']);
visual.disconnected();
visual.connected('next-token');
visual.handle({ type: 'visual_state', generation });
visual.streamStarted();
assert.equal(uploads, 1, 'Stop/Start must reuse static tags');
assert.deepEqual(visual.startPayload().visual_tags, ['calm', 'piano']);
visual.elements.ratio.value = '1:4';
visual.configure();
visual.render();
assert.equal(messages.at(-1).ratio, '1:4');
assert(visual.elements.share.textContent.includes('VLM群 80.0%'));
assert.equal(uploads, 1);

visual.elements.ratio.value = '0:1';
visual.configure();
visual.render();
assert.equal(messages.at(-1).ratio, '0:1');
assert(visual.elements.share.textContent.includes('通常群 0.0% / VLM群 100.0%'));
const extraTags = ['calm', 'piano', 'ambient', 'slow', 'soft', 'warm'];
const extraControl = visual.elements['allow-extra-tags'];
extraControl.checked = true;
extraControl.listeners.change();
assert.equal(messages.at(-1).allow_extra_tags, true);
visual.handle({ type: 'visual_applied', generation, tags: extraTags, elapsed_ms: 20 });
assert.deepEqual(visual.tags, extraTags);
assert.equal(visual.startPayload().visual.allow_extra_tags, true);
assert(visual.elements.share.textContent.includes('各タグ 16.7%'));
extraControl.checked = false;
extraControl.listeners.change();
assert.equal(messages.at(-1).allow_extra_tags, false);
assert.deepEqual(visual.tags, []);
assert(visual.elements.share.textContent.includes('通常群 0% / VLM群 0%'));
assert(visual.elements.share.textContent.includes('通常テキストは使用しません'));
visual.handle({ type: 'visual_applied', generation, tags: ['a', 'b'], elapsed_ms: 20 });
visual.handle({ type: 'visual_state', generation, tags_cleared: true });
assert.deepEqual(visual.tags, [], 'server policy removal must be reflected in the UI');
assert.equal(uploads, 1, 'changing acceptance settings must not rerun a still image');
normalHasWeight = false;
visual.elements.ratio.value = '0:1';
visual.elements.ratio.listeners.change();
const ratioMessages = messages.length;
const ratioPrepares = normalRatioPrepares;
visual.elements.ratio.value = '2:1';
visual.elements.ratio.listeners.change();
assert.equal(visual.elements.ratio.value, '0:1', 'zero normal weights must retain VLM-only ratio');
assert.equal(messages.length, ratioMessages, 'invalid normal ratio must not reach the server');
assert.equal(normalRatioPrepares, ratioPrepares);
assert(visual.elements.status.textContent.includes('0%より大きく'));
normalHasWeight = true;
visual.elements.ratio.value = '2:1';
visual.elements.ratio.listeners.change();
assert.equal(messages.at(-1).ratio, '2:1');
assert.equal(normalRatioPrepares, ratioPrepares + 1, 'normal weights must be flushed before switching');
visual.disable();
assert.deepEqual(visual.tags, []);
assert.equal(visual.enabled, false);
assert.equal(visual.elements.share.textContent, '通常群 100% / VLM群 0%');

// Auto waits for server completion, including when switching from fixed timing.
document.createElement = () => ({
  getContext: () => ({ drawImage() {} }),
  toBlob: (done) => done(new Blob(['snapshot'], { type: 'image/jpeg' })),
});
Object.assign(visual.elements.video, { videoWidth: 640, videoHeight: 480, readyState: 2 });
visual.elements.mode.value = 'camera';
visual.enabled = true;
visual.acknowledged = visual.generation;
visual.lease = { release() {} };
await visual.capture();
const waiting = visual.sequence;
const fixedUploads = uploads;
visual.elements.timing.value = 'auto';
visual.elements.timing.listeners.change();
assert.equal(visual.elements.interval.disabled, true);
await new Promise((resolve) => setTimeout(resolve, 20));
assert.equal(uploads, fixedUploads, 'switch to auto must wait for the outstanding inference');
visual.handle({ type: 'visual_state', generation: visual.generation });
await new Promise((resolve) => setTimeout(resolve, 20));
assert.equal(uploads, fixedUploads, 'config acknowledgments are not inference completion');
visual.handle({ type: 'visual_applied', generation: visual.generation, sequence: waiting,
  tags: ['calm', 'piano'], elapsed_ms: 20 });
const deadline = Date.now() + 1000;
while (uploads === fixedUploads && Date.now() < deadline) {
  await new Promise((resolve) => setTimeout(resolve, 1));
}
assert.equal(uploads, fixedUploads + 1, 'completion starts exactly one new snapshot');
await new Promise((resolve) => setTimeout(resolve, 20));
assert.equal(uploads, fixedUploads + 1, 'HTTP acceptance alone must not restart inference');
visual.handle({ type: 'visual_error', generation: visual.generation,
  sequence: visual.sequence, message: 'unavailable' });
assert.equal(visual.awaitingSequence, null);
assert(visual.nextCaptureAt > Date.now(), 'errors back off before retry');
visual.disable();
await new Promise((resolve) => setTimeout(resolve, 20));
assert.equal(uploads, fixedUploads + 1, 'disable cancels auto scheduling');
console.log('visual input state and shared camera: ok');
