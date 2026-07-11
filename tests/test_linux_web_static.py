# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Static consistency checks for the dependency-free browser client."""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "examples" / "linux_web"


class LinuxWebStaticTest(unittest.TestCase):

  def test_javascript_element_bindings_exist_in_html(self):
    html = (WEB_ROOT / "index.html").read_text()
    javascript = (WEB_ROOT / "app.js").read_text()
    element_ids = re.findall(r"document\.querySelector\('#([^']+)'\)", javascript)
    self.assertGreater(len(element_ids), 40)
    for element_id in element_ids:
      with self.subTest(element_id=element_id):
        self.assertIn(f'id="{element_id}"', html)

  def test_static_asset_cache_versions_match(self):
    html = (WEB_ROOT / "index.html").read_text()
    javascript = (WEB_ROOT / "app.js").read_text()
    self.assertIn("/static/style.css?v=9", html)
    self.assertIn("/static/app.js?v=9", html)
    self.assertIn("./hand-control.js?v=9", javascript)
    self.assertIn("./modulation.js?v=9", javascript)
    self.assertIn("/static/audio-worklet.js?v=9", javascript)
    self.assertIn("message.protocol_version < 4", javascript)


if __name__ == "__main__":
  unittest.main()
