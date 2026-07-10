# MRT2 Linux/JAX browser stream

This diagnostic client receives stateful MRT2 audio from a Linux/NVIDIA JAX
process. The server uses a three-frame (120 ms) ring buffer by default. The
AudioWorklet independently primes three frames to absorb WebSocket and browser
scheduling jitter.

Install the optional server dependencies and start the process on the GPU host:

```sh
uv sync --extra jax --extra realtime
uv run python scripts/jax_realtime_web.py --model mrt2_small
```

Keep the server bound to `127.0.0.1`. From the local computer, open a second
terminal and create an SSH tunnel:

```sh
ssh -N -L 8000:127.0.0.1:8000 USER@GPU_SERVER
```

Then open <http://127.0.0.1:8000> in Chrome or Edge. Press **Start**, allow the
browser to start Web Audio, and apply new prompts while audio is playing. A
prompt change swaps the conditioning at the next generated frame without
resetting recurrent state.

Only one WebSocket stream may own the model at a time. This is deliberate: two
concurrent generation loops would compete for the same GPU and invalidate the
real-time latency measurements.
