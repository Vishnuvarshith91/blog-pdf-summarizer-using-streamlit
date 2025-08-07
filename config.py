"""
Configuration settings for the Blog & PDF Summarizer application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class Config:
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2000))

    # Temporary file storage
    TEMP_DIR = os.getenv('TEMP_DIR', './data/temp')
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    # Allowed file types
    ALLOWED_EXTENSIONS = {'.pdf', '.txt'}

    # Logging config
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = './data/logs'

    # UI settings
    APP_TITLE = "AI Blog & PDF Summarizer 🤖"
    APP_DESCRIPTION = "Powered by Ollama - Summarize Smarter, Read Faster"

    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
