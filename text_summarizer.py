"""
Text summarization combining Ollama LLM API and extractive summarization (Sumy).
"""
import logging
from typing import Dict, Any

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from app.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.stemmer = Stemmer("english")
        self.stop_words = get_stop_words("english")

    def summarize(self, text: str, method: str = "ollama", length: str = "medium") -> Dict[str, Any]:
        if not text or len(text.strip()) < 100:
            raise ValueError("Text too short for summarization")

        sentence_counts = {"short": 3, "medium":5, "long": 8}
        sentences = sentence_counts.get(length, 5)
        logger.info(f"Summarizing with method={method}, length={length} ({sentences} sentences)")

        if method == "ollama":
            summary = self._ollama_summarize(text, length)
        elif method == "extractive":
            summary = self._extractive_summarize(text, sentences)
        elif method == "hybrid":
            summary = self._hybrid_summarize(text, sentences, length)
        else:
            raise ValueError(f"Unsupported summarization method: {method}")

        original_words = len(text.split())
        summary_words = len(summary.split())
        compression = (original_words - summary_words)/original_words * 100

        return {
            "summary": summary,
            "method": method,
            "length": length,
            "original_words": original_words,
            "summary_words": summary_words,
            "compression_ratio": round(compression, 2),
            "sentences": len([s for s in summary.split('.') if s.strip()])
        }

    def _ollama_summarize(self, text: str, length: str) -> str:
        if not self.ollama_client.is_available():
            raise RuntimeError("Ollama service not available")
        length_prompts = {
            "short": "Create a brief summary in 2-3 sentences.",
            "medium": "Create a detailed summary in 4-6 sentences covering main points.",
            "long": "Create an in-depth summary in 7-10 sentences covering all key aspects."
        }
        prompt = f"""
        {length_prompts[length]}

        Please summarize the following text:

        {text}

        Summary:
        """
        return self.ollama_client.generate_summary(text, prompt)

    def _extractive_summarize(self, text: str, sentence_count: int) -> str:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LuhnSummarizer(self.stemmer)
            summarizer.stop_words = self.stop_words
            summary_sentences = summarizer(parser.document, sentence_count)
            return ' '.join(str(s) for s in summary_sentences)
        except Exception as e:
            logger.error(f"Extractive summarization error: {e}")
            return ""

    def _hybrid_summarize(self, text: str, sentence_count: int, length: str) -> str:
        try:
            extractive = self._extractive_summarize(text, sentence_count * 2)
            if self.ollama_client.is_available():
                prompt = f"""
                Please refine and improve this extractive summary, making it more coherent and natural:

                Extractive Summary: {extractive}

                Original Text (truncated): {text[:2000]}...

                Improved Summary:
                """
                return self.ollama_client.generate_summary(extractive, prompt)
            else:
                return extractive
        except Exception as e:
            logger.warning(f"Hybrid summarization fallback due to error: {e}")
            return self._extractive_summarize(text, sentence_count)
