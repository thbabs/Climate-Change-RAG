"""Check that generate_answer_gemini passes the configured model name to the SDK."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

os.environ["GOOGLE_API_KEY"] = "test-key-not-used"
os.environ.pop("GEMINI_MODEL", None)

# Skip embedding download and Chroma startup; this check only covers generation.
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("chromadb", MagicMock())

import app  # noqa: E402


def _fake_model():
    response = MagicMock()
    response.text = "ok"
    model = MagicMock()
    model.generate_content.return_value = response
    return model


class GeminiModelTests(unittest.TestCase):
    def test_default_model_is_passed_to_generative_model(self):
        os.environ.pop("GEMINI_MODEL", None)
        with patch.object(app.genai, "GenerativeModel", return_value=_fake_model()) as ctor:
            result = app.generate_answer_gemini(
                "what is climate change?",
                [{"text": "context"}],
            )
        ctor.assert_called_once_with("gemini-3.8-flash")
        self.assertEqual(result, "ok")

    def test_gemini_model_env_overrides_the_name(self):
        os.environ["GEMINI_MODEL"] = "gemini-3.6-flash"
        try:
            with patch.object(app.genai, "GenerativeModel", return_value=_fake_model()) as ctor:
                app.generate_answer_gemini("q", [{"text": "context"}])
            ctor.assert_called_once_with("gemini-3.6-flash")
        finally:
            os.environ.pop("GEMINI_MODEL", None)

    def test_installed_sdk_accepts_stable_flash_model_name(self):
        model = app.genai.GenerativeModel("gemini-3.8-flash")
        self.assertIn("gemini-3.8-flash", model.model_name)


if __name__ == "__main__":
    unittest.main()
