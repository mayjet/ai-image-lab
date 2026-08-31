"""Low-memory policy tests that do not load torch or model weights."""

import unittest
from pathlib import Path

from flux2.generate import select_memory_mode, select_quantization
from flux2.prompt_markdown import FluxMarkdownPrompt, resolve_references


class MemoryPolicyTest(unittest.TestCase):
    def test_x86_8gb_uses_strongest_cpu_offload(self):
        self.assertEqual(select_memory_mode("auto", 8.0, 28.0), "sequential-offload")
        self.assertEqual(select_quantization("auto", 8.0), "bnb4")

    def test_high_vram_stays_on_gpu_without_quantization(self):
        self.assertEqual(select_memory_mode("auto", 16.0, 8.0), "gpu")
        self.assertEqual(select_quantization("auto", 16.0), "none")

    def test_low_available_cpu_ram_is_rejected(self):
        with self.assertRaises(RuntimeError):
            select_memory_mode("auto", 8.0, 19.9)

    def test_old_offload_name_remains_compatible(self):
        self.assertEqual(select_memory_mode("offload", 8.0, 28.0), "model-offload")

    def test_zero_references_does_not_touch_configured_paths(self):
        document = FluxMarkdownPrompt(
            path=Path("prompt.md"),
            settings={},
            references={"source_image": "missing.png"},
            lora={},
            generation_preferences="",
            positive="test",
            negative="",
            rules="",
        )
        self.assertEqual(resolve_references(document, Path.cwd(), max_images=0), [])


if __name__ == "__main__":
    unittest.main()
