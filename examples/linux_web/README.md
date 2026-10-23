# MRT2 Linux/JAX browser stream

This diagnostic client receives stateful MRT2 audio from a Linux/NVIDIA JAX
process. The server uses a three-frame (120 ms) ring buffer by default. The
AudioWorklet independently primes three frames to absorb WebSocket and browser
scheduling jitter.

Install only the optional server packages into the existing GPU-enabled uv
environment, then start the process on the GPU host:

```sh
source .venv/bin/activate
uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.34"
python scripts/jax_realtime_web.py --model mrt2_small
```

The WebSocket server enables Apple live parity mode by default. It uses the
same live-app sampling defaults, masks the final 6 of 12 MusicCoCa RVQ
conditioning levels, runs the same two classifier-free-guidance branches
(MusicCoCa and notes), and streams the decoder's float32 output without the
legacy JAX int16 conversion. For an A/B comparison with the previous Linux
path, start it with `--no-apple-live-parity` instead. The two CFG branches
increase GPU work, so re-check the 40 ms frame deadline after enabling this
mode. Sampling remains stochastic, so Apple/MLX and Linux/JAX do not produce
sample-identical audio.

Keep the server bound to `127.0.0.1`. From the local computer, open a second
terminal and create an SSH tunnel:

```sh
ssh -N -L 8000:127.0.0.1:8000 USER@GPU_SERVER
```

Then open <http://127.0.0.1:8000> in Chrome or Edge. Press **Start** and allow
the browser to start Web Audio. Up to 32 prompt texts are embedded once and
cached. Slider changes blend the cached embeddings immediately; text edits are
encoded only after pressing **Encode text changes**. The 2D surface uses
normalized inverse-square-distance weights. None of these controls resets the
recurrent streaming state.

The native MLX engine stores six prompt slots, but the Linux/JAX path blends
MusicCoCa embeddings before tokenization and therefore does not change the
model input shape when more prompts are added. The browser and WebSocket retain
a 32-prompt safety limit to bound text-encoding work and request size. New 2D
nodes use a golden-angle layout to reduce overlap beyond the first six.

The browser buffer starts at three frames. If an AudioWorklet underrun occurs,
it automatically increases its target by one frame, up to six frames. This
trades 40 ms of additional control latency for more scheduling-jitter margin.

## Recording and WAV export

Recording is performed on the GPU server from the exact frames sent to the
browser. Press **Record** before playback to arm recording from the first
streamed frame, or press it during playback to begin on the next 40 ms frame.
Press **Stop Recording** to finalize the file while playback continues. Pressing
the main **Stop** button also finalizes an active recording before the WebSocket
is closed.

Files are written under `outputs/` as
`mrt2_recording_YYYYMMDD_HHMMSS_mmm.wav`. The format is 48 kHz, stereo, signed
16-bit PCM for broad player and DAW compatibility. The browser shows the saved
relative path and duration, and the backend logs the same path. Empty armed
recordings are removed. `outputs/` is already excluded by `.gitignore`.

## Web MIDI

Open the page in local Chrome or Edge through the SSH tunnel, then press
**Enable MIDI** and grant browser permission. Select a physical MIDI input, or
enable **PC keyboard** for testing without hardware. MIDI note-on/off events
are latched on the server so even a note shorter than one 40 ms inference frame
is observed for one frame.

- **Auto-Strum on** sends token 3 while a note is held, allowing the model to
  retrigger, bow, strum, or arpeggiate it.
- **Auto-Strum off** sends an onset token followed by continuation tokens.
- Sustain pedal CC64 and All Notes Off CC120/CC123 are supported in the browser.

The PC keyboard mapping follows the official app: A/W/S/E/D/F/T/G/Y/H/U/J
starts at C3, K/O/L/P/; continues above it, and Z/X changes octave.

### Jam / Solo performance mode

This switch follows the dedicated native Jam app rather than the simpler Solo
toggle in the main plugin:

- **Jam** uses the visible prompt text unchanged and an unmask width of zero.
  Held notes are explicit while the rest remain masked, allowing accompaniment.
- **Solo** re-encodes every active text as `SOLO <prompt>`, uses an unmask width
  of 127 so every non-held note is explicitly off, and gates the outgoing audio
  after note release with the native two-second linear decay. During the same
  release the Note CFG value ramps from its configured value toward 50.
- Solo playback stops after 30 seconds without MIDI note activity, matching the
  Jam app. Switching modes preserves recurrent streaming state; the first
  switch in each direction may run MusicCoCa encoding, after which cached
  embeddings are reused.

## Drum conditioning

**No Drums** matches the native app's drumless switch. Off leaves the single
drum-conditioning token masked (`-1`), allowing the model to decide. On sends
token `0` every 40 ms frame, encouraging drum-free generation. Switching does
not reset the recurrent streaming state or re-encode prompts.

## Sampling controls

The Global Generation panel exposes the native MRT2 parameter ranges:

- **Temperature**: 0.0–3.0 in 0.05 steps. Lower values are more stable;
  higher values introduce more sampling variation.
- **Top-K Sampling**: 1–1024. This limits the candidate set considered for
  each sampled token.

The browser sends both values atomically and at most once every 40 ms. The JAX
step receives arrays with unchanged shape and dtype, so live changes preserve
the recurrent state and do not require prompt encoding or recompilation.

## Prompt modulation

The browser provides two lightweight modulation sources. Both reuse the
existing cached prompt-weight WebSocket message and do not add JAX work:

- **LFO** supports sine, triangle, square, and saw waves from 0.05–5 Hz. Pick a
  target prompt and a minimum/maximum raw weight. The LFO may be enabled before
  playback; the current local weights are included in the initial prompt bank.
- **MIDI CC** maps CC values 0–127 to a selected prompt weight. Enter a
  controller number from 0–119 or press **Learn** and move a physical control.
  The mapping can be inverted. It listens to the Web MIDI input already
  selected in the Note Conditioning panel.

If Hand, LFO, and MIDI CC address the same prompt at the same time, the most
recent update wins. Use separate target prompts when combining sources for
predictable modulation.

## Hand landmark and gesture control

The hand controller runs entirely in the local browser and does not upload
camera frames to the GPU server. It has three modes:

- **Pinch** maps the normalized thumb/index distance to one prompt,
  Temperature, or Top-K. A closed pinch is 0% and a wide spread is 100%;
  **Invert** reverses the mapping.
- **Gesture Map** assigns Victory/peace, Open Palm, Fox, and Closed Fist to
  prompt slots or either sampling control. Sampling controls use recognition
  confidence as their normalized knob position. MediaPipe's canned recognizer
  handles all but Fox; Fox is classified from finger extension and
  thumb-contact landmarks.
- **Hybrid** assigns Pinch, Gesture Map, or Off independently to the anatomical
  right and left hands. It defaults to right-hand Gesture Map plus left-hand
  Pinch. Pinch sets its prompt's share and active gestures blend the remaining
  share; the roles can be swapped when camera handedness is inconvenient.

Gesture Map tracks up to two hands. Two different simultaneous gestures blend
their assigned prompts using smoothed recognition confidence. Recognition must
persist for two frames before switching, and the last recognized gesture is
held for 320 ms through brief tracking loss. Detection is limited to about
15 fps, independently of the audio-generation loop.

In Hybrid mode, avoid assigning the Pinch target and every gesture to the same
prompt: that is mathematically a 100% single-prompt mix and the Pinch movement
will not be audible. The camera preview is mirrored, while the left/right labels
refer to the performer's anatomical hands.

The first launch downloads the pinned MediaPipe Tasks Vision runtime and the
official Gesture Recognizer model in the browser. Camera access requires
a secure browser context; `http://127.0.0.1` through the SSH tunnel qualifies,
but a plain remote-server HTTP address usually does not. Chrome or Edge is
recommended. Camera inference is local and independent of the JAX/CUDA audio
generation thread.

Only one WebSocket stream may own the model at a time. This is deliberate: two
concurrent generation loops would compete for the same GPU and invalidate the
real-time latency measurements.

## Visual music input

The **画像から音楽スタイルを追加** panel adds a separate VLM tag group.
The default model is **Gemma 4 E4B** (`gemma4:e4b`) served by the existing
Ollama instance at `http://127.0.0.1:21434`. Qwen3.5-4B (`qwen3.5:4b`) and
Gemini (`gemini-3.5-flash-lite`) are selectable in every input mode.

Install the optional dependencies in the existing MRT2 environment:

```sh
uv pip install -e ".[realtime,visual,gemini]"
python scripts/jax_realtime_web.py --model mrt2_small --port 8001
```

Start Ollama separately using your existing `ollama-local/env.sh` environment
and `ollama_local serve`. This application does not start Ollama or download
models. `--ollama-host http://127.0.0.1:21434` can override its address. Gemini
requires `GEMINI_API_KEY` in the **server's** environment; no key is sent to the
browser. Selecting Gemini sends the image to Google's API; Ollama sends it
only to the configured Ollama host.

1. Choose **静止画像**, **画面共有**, or **カメラ** and the VLM model.
2. For a still image, select a JPEG, PNG, WebP, or BMP file (up to 8 MiB and
   24 megapixels). For live sources, choose fixed snapshot intervals (default
   3 s) or **自動** (start the next snapshot after inference and tag application
   finish). Auto submits only one image at a time; errors retry after 1 s.
3. Press **画像入力を開始 / 再実行**. Screen sharing and camera access request
   browser permission. Inputs prepared before playback are inferred when
   playback starts. Screen sharing requires a button click and a secure
   browser context, such as localhost through the SSH tunnel.
4. Press the existing **再生開始** button if playback is stopped.

The three sources are exclusive. Selecting another mode/model disables the
previous source; press the visual start button to activate the new selection.
The camera can be shared with the existing hand/gesture controller. Stopping
either consumer does not stop the other's camera stream.

Still images run once per explicit visual start. Their tags survive audio
Stop/Start within the same page, without another VLM request. Live capture
stops on audio Stop/disconnect or when the screen/camera track ends; start
the visual input again to resume. **画像入力を無効化** clears the visual tags.

The ordinary 32-prompt limit, sliders, 2D Surface, LFO, MIDI CC and hand
targets apply only to the normal group. Their percentages are **within that
group**. With normal:VLM set to 2:1, the normal group's mixed embedding receives
2/3 of the total coefficient. Each of m visual tags receives 1/(3m).
Ratios use integers 1–4 on either side, with equivalent ratios shown once:
4:1, 3:1, 2:1 (default), 3:2, 4:3, 1:1, 3:4, 2:3, 1:2, 1:3, and 1:4.
**0:1 (VLM only)** excludes normal text from the generated style. Normal text
controls remain available, but do not affect the style until a nonzero normal
ratio is selected again. While no VLM result is available, including when
visual input is disabled, 0:1 uses masked style conditioning instead of falling
back to normal text. Previously buffered audio still plays normally.
At 0:1 all normal sliders may be 0%, both at playback start and during playback.
Their raw and displayed normalized weights remain zero. To return to a ratio
that uses normal text, first raise at least one normal slider above 0%; an
attempted ratio change with all normal weights at zero keeps 0:1 and displays
an explanation. Normal-text ratios still reject an all-zero normal group.
For example, 1:4 allocates 20% to normal text and 80% to the VLM group.
Tags are individually
embedded with the same mapper/seed as normal prompts, then averaged before
the groups are mixed. The final embedding goes through the existing RVQ and
JAX generation path. Equal coefficients do not guarantee equal audible effects.
Solo uses the same `SOLO ` prefix for each visual tag without re-running VLM.

No visual result means normal conditioning at 100%, except for 0:1 as above.
Failures keep the last
successful result from the same mode/model; selecting a new source/model
clears that result until a new one succeeds. The prompt asks for 3–5 tags,
never more than 5. **5個を超えるタグも使用する** is off by default: an output
containing more than 5 nonempty comma fields is rejected in full. Turn it on
to accept all returned tags, including the sixth and later tags, with equal
weight within the VLM group. The instruction still requests 3–5 tags. Both
settings keep the existing 1024-character output and 128-character per-tag
limits, remove empty comma fields and duplicates, and reject multiline output.
The setting is checked again before publishing a style, so changing it during
inference or embedding cannot apply a result that the new setting rejects.
Turning it off also removes an already applied oversized result; in 0:1 this
returns to masked style conditioning. For a still image, press visual start
again to obtain a new result. Cached static tags respect the setting on restart.

Visual work is asynchronous. At most one image is processed plus one latest
pending image per stream. Outdated generations cannot apply their results.
The server serializes MusicCoCa text/mapper calls through a shared worker
with a 256-entry cache. Normal text encoding no longer stalls MIDI/control
reception. Images are sent as bounded HTTP bodies, and progress/results use
additional JSON messages on the existing WebSocket. The PCM transport,
recurrent JAX state, and audio buffers are unchanged.

Local VLM and MRT2 may still compete for GPU resources. Check the existing
generation time, deadline-miss and underrun metrics during simultaneous use;
standalone VLM benchmark times are not end-to-end music response times.
Remote inference has a 60-second deadline. A cancelled HTTP call does not
guarantee the Ollama server has already stopped GPU work.

Dependency-light checks (no model weights/API calls):

```sh
python -m unittest discover -s tests -p 'test_visual*.py' -v
node --experimental-default-type=module tests/visual_input_test.mjs
```
