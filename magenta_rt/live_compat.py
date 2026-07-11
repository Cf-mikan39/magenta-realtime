# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-light settings shared with the native Apple live engine."""


APPLE_LIVE_MUSICCOCA_MASKED_TAIL_LEVELS = 6


def mask_musiccoca_tail(
    tokens: list[int], masked_tail_levels: int
) -> list[int]:
  """Return MusicCoCa tokens with the requested fine RVQ tail masked.

  The native Apple live engine retains all 12 tokens for prompt blending, but
  replaces the final six conditioning levels with ``-1`` immediately before
  generation. Keeping this operation after embedding/quantization means prompt
  blending still operates at full resolution, matching that behavior.
  """
  if isinstance(masked_tail_levels, bool) or not isinstance(
      masked_tail_levels, int
  ):
    raise TypeError('masked_tail_levels must be an integer')
  if not 0 <= masked_tail_levels <= len(tokens):
    raise ValueError(
        'masked_tail_levels must be between 0 and the token count'
    )
  result = list(tokens)
  if masked_tail_levels:
    result[-masked_tail_levels:] = [-1] * masked_tail_levels
  return result
