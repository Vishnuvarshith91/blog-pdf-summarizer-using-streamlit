"""
Ollama client interface for generating summaries using the local LLM.
"""
import logging
import ollama
from typing import Optional
from app.config import Config

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.model = Config.OLLAMA_MODEL
        self.max_tokens = Config.MAX_TOKENS

    def is_available(self) -> bool:
        try:
            available_models = ollama.list()
            return True
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False

    def generate_summary(self, text: str, custom_prompt: Optional[str] = None) -> str:
        """
        Generates a summary from text using the Ollama model.
        """
        try:
            if not custom_prompt:
                custom_prompt = (
                    "Please provide a detailed summary of the following content:\n\n"
                    "{text}\n\nSummary:"
                )
            prompt = custom_prompt.format(text=text[:8000])  # limit input length

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0.7, 'top_p': 0.9, 'max_tokens': self.max_tokens}
            )
            summary = response['response'].strip()
            logger.info(f"Summary generated: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary via Ollama: {e}")
            raise RuntimeError(f"Ollama summarization error: {e}")

    def get_available_models(self) -> list:
        try:
            models = ollama.list()
            return [m['name'] for m in models.get('models', [])]
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
            return []
