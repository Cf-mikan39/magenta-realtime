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

Only one WebSocket stream may own the model at a time. This is deliberate: two
concurrent generation loops would compete for the same GPU and invalidate the
real-time latency measurements.
