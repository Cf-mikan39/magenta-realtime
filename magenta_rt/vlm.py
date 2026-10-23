"""Image-to-music backends. No JAX import or model state access."""

import base64
from io import BytesIO
import os


MODELS = {
    "gemma": ("ollama", "gemma4:e4b"),
    "qwen": ("ollama", "qwen3.5:4b"),
    "gemini": ("gemini", "gemini-3.5-flash-lite"),
}
DEFAULT_MODEL = "gemma"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TAGS = 5
INSTRUCTION = """You are a visual-to-music style translator.
Analyze the input image and convert its visual atmosphere into an English
music style prompt for an instrumental music generation model.
Return exactly one line of comma-separated music tags. Use 3 to 5 tags, never more than 5.
Use individual words or short noun phrases, not complete sentences.
Do not use conjunctions such as "and" or "with".
Do not include explanations or introductory text.
Focus on mood, genre, instruments, musical texture, rhythm, tempo, and energy.
Avoid literal image descriptions. Do not add environmental sound effects.
Example: dreamy, ambient lo-fi, soft piano, warm synth pads, low energy"""


def parse_music_tags(text, *, allow_extra_tags=False):
  if not isinstance(allow_extra_tags, bool):
    raise ValueError("allow_extra_tags must be a boolean")
  if not isinstance(text, str) or not text.strip() or len(text) > 1024:
    raise ValueError("VLM must return a non-empty music-tag line (max 1024 characters)")
  if "\n" in text.strip() or "\r" in text.strip():
    raise ValueError("VLM returned multiple lines")
  fields = [t.strip().lower() for t in text.split(",") if t.strip()]
  tags = tuple(dict.fromkeys(fields))
  if not fields or any(len(t) > 128 for t in tags):
    raise ValueError("VLM must return non-empty music tags (max 128 characters per tag)")
  if len(fields) > MAX_TAGS and not allow_extra_tags:
    raise ValueError(
        f"VLM returned {len(fields)} tags; enable '5個を超えるタグも使用する' "
        f"to accept more than {MAX_TAGS} tags"
    )
  return tags


def prepare_image(data):
  from PIL import Image, ImageOps
  if not data or len(data) > MAX_IMAGE_BYTES:
    raise ValueError("image must be between 1 byte and 8 MiB")
  with Image.open(BytesIO(data)) as source:
    if source.width * source.height > 24_000_000:
      raise ValueError("image exceeds 24 megapixels")
    if source.format not in ("JPEG", "PNG", "WEBP", "BMP"):
      raise ValueError("use a JPEG, PNG, WebP or BMP image")
    image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((1024, 1024))
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


class VlmBackend:
  """One reusable async client for each selected model."""

  def __init__(self, model=DEFAULT_MODEL, host="http://127.0.0.1:21434", timeout=60):
    if model not in MODELS:
      raise ValueError("unknown VLM model")
    self.provider, self.model = MODELS[model]
    self.timeout = timeout
    if self.provider == "ollama":
      import httpx
      self.client = httpx.AsyncClient(base_url=host, timeout=timeout)
    else:
      from google import genai
      key = os.environ.get("GEMINI_API_KEY")
      if not key:
        raise ValueError("GEMINI_API_KEY is not set on the server")
      self.client = genai.Client(api_key=key).aio

  async def infer(self, jpeg):
    if self.provider == "ollama":
      response = await self.client.post("/api/chat", json={
          "model": self.model, "stream": False, "think": False,
          "keep_alive": "5m",
          "messages": [{"role": "user", "content": INSTRUCTION,
                        "images": [base64.b64encode(jpeg).decode("ascii")]}],
          "options": {"temperature": 0, "seed": 42, "num_predict": 256},
      })
      response.raise_for_status()
      result = response.json()
      if result.get("done_reason") == "length":
        raise ValueError("VLM output was truncated")
      return result.get("message", {}).get("content", "")
    from google.genai import types
    response = await self.client.models.generate_content(
        model=self.model,
        contents=[INSTRUCTION, types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
        config=types.GenerateContentConfig(
            temperature=0, seed=42, max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    candidates = getattr(response, "candidates", None) or []
    if candidates and "MAX_TOKENS" in str(candidates[0].finish_reason):
      raise ValueError("VLM output was truncated")
    return response.text or ""

  async def close(self):
    await self.client.aclose()
