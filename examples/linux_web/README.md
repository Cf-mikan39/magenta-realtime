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
the browser to start Web Audio. Up to six prompt texts are embedded once and
cached. Slider changes blend the cached embeddings immediately; text edits are
encoded only after pressing **Encode text changes**. The 2D surface uses
normalized inverse-square-distance weights. None of these controls resets the
recurrent streaming state.

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
- **Solo off** unmasks four neighboring pitches and lets MRT2 add
  accompaniment.
- **Solo on** explicitly marks every non-held pitch off.
- Sustain pedal CC64 and All Notes Off CC120/CC123 are supported in the browser.

The PC keyboard mapping follows the official app: A/W/S/E/D/F/T/G/Y/H/U/J
starts at C3, K/O/L/P/; continues above it, and Z/X changes octave.

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
