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

Only one WebSocket stream may own the model at a time. This is deliberate: two
concurrent generation loops would compete for the same GPU and invalidate the
real-time latency measurements.
