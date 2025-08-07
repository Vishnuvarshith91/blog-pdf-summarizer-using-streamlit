# Complete Step-by-Step Development Guide
## Building an AI Blog & PDF Summarizer with Python, VS Code, and Ollama

This comprehensive guide will walk you through creating a modern, Gen Z-inspired web application for summarizing blogs and PDFs using Ollama as your local AI engine.

---

## 📋 Prerequisites

Before we begin, ensure you have:
- **Python 3.8+** installed on your system
- **VS Code** with Python extension
- **Git** for version control
- **At least 8GB RAM** (16GB+ recommended for Ollama)
- **Stable internet connection** for initial setup

---

## 🚀 Phase 1: Environment Setup

### Step 1: Install Ollama

**For macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
# Or using Homebrew
brew install ollama
```

**For Windows:**
1. Download installer from [ollama.ai](https://ollama.ai)
2. Run the installer and follow setup wizard
3. Verify installation: `ollama --version`

**For Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Setup Ollama Model
```bash
# Start Ollama service
ollama serve

# In a new terminal, pull a model (choose one)
ollama pull llama2          # 3.8GB - Good for basic tasks
ollama pull llama2:13b      # 7.3GB - Better performance
ollama pull mistral         # 4.1GB - Fast and efficient
ollama pull codellama       # 3.8GB - Code-focused

# Test the installation
ollama run llama2
>>> Hello, how are you?
```

### Step 3: Create Project Directory
```bash
# Create and navigate to project directory
mkdir blog-pdf-summarizer
cd blog-pdf-summarizer

# Initialize Git repository
git init
```

### Step 4: Setup Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 5: Install VS Code Extensions

Open VS Code and install these essential extensions:

1. **Python** by Microsoft
2. **Pylance** by Microsoft  
3. **Python Docstring Generator**
4. **GitLens — Git supercharged**
5. **Better Comments**
6. **Error Lens**
7. **autoDocstring - Python Docstring Generator**
8. **Python Indent**

---

## 🏗️ Phase 2: Project Structure Setup

### Step 1: Create Directory Structure
```bash
# Create main directories
mkdir -p app/utils
mkdir -p static/css static/images
mkdir -p data/temp data/logs
mkdir -p tests
mkdir -p docs

# Create __init__.py files
touch app/__init__.py
touch app/utils/__init__.py
touch tests/__init__.py
```

### Step 2: Create Core Configuration Files

**Create requirements.txt:**
```txt
streamlit==1.31.0
ollama==0.1.8
requests==2.31.0
beautifulsoup4==4.12.2
PyPDF2==3.0.1
pymupdf==1.23.21
python-dotenv==1.0.0
validators==0.22.0
nltk==3.8.1
sumy==0.11.0
lxml==4.9.3
html5lib==1.1
Pillow==10.1.0
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Create .env file:**
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
MAX_TOKENS=2000
TEMP_DIR=./data/temp
LOG_LEVEL=INFO
```

**Create .gitignore:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# VS Code
.vscode/settings.json
.vscode/launch.json

# Environment variables
.env
.env.local
.env.production

# Temporary files
data/temp/*
data/logs/*
*.log

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
```

### Step 3: VS Code Configuration

**Create .vscode/settings.json:**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true
    },
    "python.autoComplete.extraPaths": ["./app"],
    "python.analysis.extraPaths": ["./app"]
}
```

**Create .vscode/launch.json:**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Streamlit App",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "app/main.py", "--server.port", "8501"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Debug Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```

---

## 💻 Phase 3: Core Application Development

### Step 1: Configuration Module (app/config.py)
```python
"""
Configuration settings for the Blog & PDF Summarizer application.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2000))
    
    # File Handling
    TEMP_DIR = os.getenv('TEMP_DIR', './data/temp')
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf', '.txt'}
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = './data/logs'
    
    # UI Configuration
    APP_TITLE = "AI Blog & PDF Summarizer 🤖"
    APP_DESCRIPTION = "Powered by Ollama - Summarize Smarter, Read Faster"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist."""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
```

### Step 2: Ollama Client (app/utils/ollama_client.py)
```python
"""
Ollama API client for handling LLM communications.
"""
import ollama
import logging
from typing import Optional, Dict, Any
from app.config import Config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.max_tokens = Config.MAX_TOKENS
        
    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            ollama.list()
            return True
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def generate_summary(self, text: str, custom_prompt: Optional[str] = None) -> str:
        """
        Generate summary using Ollama.
        
        Args:
            text: Text to summarize
            custom_prompt: Optional custom prompt template
            
        Returns:
            Generated summary
        """
        try:
            # Default summarization prompt
            if not custom_prompt:
                custom_prompt = """
                Please provide a comprehensive summary of the following text. 
                Focus on the main points, key insights, and important details. 
                Make the summary clear, concise, and well-structured:
                
                {text}
                
                Summary:
                """
            
            # Format prompt with text
            formatted_prompt = custom_prompt.format(text=text[:8000])  # Limit input size
            
            # Generate response
            response = ollama.generate(
                model=self.model,
                prompt=formatted_prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': self.max_tokens
                }
            )
            
            summary = response['response'].strip()
            logger.info(f"Successfully generated summary of length: {len(summary)}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise Exception(f"Failed to generate summary: {str(e)}")
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
```

### Step 3: Blog Scraper (app/utils/blog_scraper.py)
```python
"""
Blog content extraction utilities.
"""
import requests
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import Dict, Optional
import validators

logger = logging.getLogger(__name__)

class BlogScraper:
    """Utility class for scraping blog content."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_content(self, url: str) -> Dict[str, str]:
        """
        Extract title and content from a blog URL.
        
        Args:
            url: Blog URL to scrape
            
        Returns:
            Dictionary with title, content, and metadata
        """
        try:
            # Validate URL
            if not validators.url(url):
                raise ValueError("Invalid URL format")
            
            logger.info(f"Scraping content from: {url}")
            
            # Fetch the webpage
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract metadata
            description = self._extract_description(soup)
            
            result = {
                'title': title,
                'content': content,
                'description': description,
                'url': url,
                'word_count': len(content.split()) if content else 0
            }
            
            logger.info(f"Successfully extracted content: {result['word_count']} words")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Network error while scraping {url}: {e}")
            raise Exception(f"Failed to fetch content from URL: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            raise Exception(f"Failed to extract content: {str(e)}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try different title sources
        title_selectors = [
            'h1',
            'title',
            '[property="og:title"]',
            '.post-title',
            '.entry-title',
            '.article-title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title and len(title) > 5:
                    return title
        
        return "Untitled Article"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try different content selectors
        content_selectors = [
            'article',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.content',
            'main',
            '[role="main"]'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator=' ').strip()
                if len(content) > 100:  # Ensure substantial content
                    return self._clean_text(content)
        
        # Fallback: extract all paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs])
        
        return self._clean_text(content) if content else ""
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description/meta description."""
        description_selectors = [
            '[property="og:description"]',
            '[name="description"]',
            '.excerpt',
            '.post-excerpt'
        ]
        
        for selector in description_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get('content') or element.get_text()
                if content and len(content.strip()) > 10:
                    return content.strip()
        
        return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
```

### Step 4: PDF Processor (app/utils/pdf_processor.py)
```python
"""
PDF processing utilities for text extraction.
"""
import os
import logging
import tempfile
from typing import Dict, BinaryIO
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from app.config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Utility class for processing PDF files."""
    
    def __init__(self):
        self.max_file_size = Config.MAX_FILE_SIZE
    
    def extract_text(self, file_buffer: BinaryIO, filename: str) -> Dict[str, str]:
        """
        Extract text from PDF file buffer.
        
        Args:
            file_buffer: PDF file buffer from Streamlit
            filename: Original filename
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            logger.info(f"Processing PDF: {filename}")
            
            # Check file size
            file_buffer.seek(0, 2)  # Go to end
            file_size = file_buffer.tell()
            file_buffer.seek(0)  # Reset to beginning
            
            if file_size > self.max_file_size:
                raise ValueError(f"File too large. Max size: {self.max_file_size/1024/1024:.1f}MB")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_buffer.read())
                temp_file_path = temp_file.name
            
            try:
                # Try PyMuPDF first (better text extraction)
                text = self._extract_with_pymupdf(temp_file_path)
                
                if not text.strip():
                    # Fallback to PyPDF2
                    file_buffer.seek(0)
                    text = self._extract_with_pypdf2(file_buffer)
                
                # Extract metadata
                metadata = self._extract_metadata(temp_file_path)
                
                result = {
                    'text': text,
                    'filename': filename,
                    'page_count': metadata.get('page_count', 0),
                    'word_count': len(text.split()) if text else 0,
                    'title': metadata.get('title', filename)
                }
                
                logger.info(f"Successfully extracted {result['word_count']} words from {result['page_count']} pages")
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _extract_with_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (fitz)."""
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def _extract_with_pypdf2(self, file_buffer: BinaryIO) -> str:
        """Extract text using PyPDF2 as fallback."""
        text = ""
        try:
            reader = PdfReader(file_buffer)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def _extract_metadata(self, file_path: str) -> Dict[str, str]:
        """Extract PDF metadata."""
        metadata = {}
        try:
            with fitz.open(file_path) as doc:
                metadata = {
                    'page_count': doc.page_count,
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'creator': doc.metadata.get('creator', '')
                }
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            
        return metadata
```

### Step 5: Text Summarizer (app/utils/text_summarizer.py)
```python
"""
Text summarization utilities combining multiple approaches.
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
    """Advanced text summarization using multiple techniques."""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.stemmer = Stemmer("english")
        self.stop_words = get_stop_words("english")
    
    def summarize(self, text: str, method: str = "ollama", length: str = "medium") -> Dict[str, Any]:
        """
        Summarize text using specified method.
        
        Args:
            text: Input text to summarize
            method: Summarization method ("ollama", "extractive", "hybrid")
            length: Summary length ("short", "medium", "long")
            
        Returns:
            Summary results with metadata
        """
        try:
            if not text or len(text.strip()) < 100:
                raise ValueError("Text too short for meaningful summarization")
            
            # Determine sentence count based on length
            sentence_counts = {
                "short": 3,
                "medium": 5,
                "long": 8
            }
            sentence_count = sentence_counts.get(length, 5)
            
            logger.info(f"Summarizing text with method: {method}, length: {length}")
            
            if method == "ollama":
                summary = self._ollama_summarize(text, length)
            elif method == "extractive":
                summary = self._extractive_summarize(text, sentence_count)
            elif method == "hybrid":
                summary = self._hybrid_summarize(text, sentence_count, length)
            else:
                raise ValueError(f"Unknown summarization method: {method}")
            
            # Calculate metrics
            original_words = len(text.split())
            summary_words = len(summary.split())
            compression_ratio = (original_words - summary_words) / original_words * 100
            
            result = {
                'summary': summary,
                'method': method,
                'length': length,
                'original_words': original_words,
                'summary_words': summary_words,
                'compression_ratio': round(compression_ratio, 2),
                'sentences': len([s for s in summary.split('.') if s.strip()])
            }
            
            logger.info(f"Summary generated: {compression_ratio:.1f}% compression")
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise Exception(f"Failed to generate summary: {str(e)}")
    
    def _ollama_summarize(self, text: str, length: str) -> str:
        """Summarize using Ollama LLM."""
        if not self.ollama_client.is_available():
            raise Exception("Ollama service not available")
        
        # Create length-specific prompt
        length_instructions = {
            "short": "Create a brief, concise summary in 2-3 sentences.",
            "medium": "Create a comprehensive summary in 4-6 sentences covering the main points.",
            "long": "Create a detailed summary in 7-10 sentences covering all important aspects."
        }
        
        prompt = f"""
        {length_instructions[length]}
        
        Please summarize the following text:
        
        {text}
        
        Summary:
        """
        
        return self.ollama_client.generate_summary(text, prompt)
    
    def _extractive_summarize(self, text: str, sentence_count: int) -> str:
        """Summarize using extractive method (Luhn algorithm)."""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LuhnSummarizer(self.stemmer)
            summarizer.stop_words = self.stop_words
            
            summary_sentences = summarizer(parser.document, sentence_count)
            summary = ' '.join([str(sentence) for sentence in summary_sentences])
            
            return summary
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            raise Exception("Extractive summarization failed")
    
    def _hybrid_summarize(self, text: str, sentence_count: int, length: str) -> str:
        """Combine extractive and Ollama methods for better results."""
        try:
            # First, get extractive summary to identify key sentences
            extractive_summary = self._extractive_summarize(text, sentence_count * 2)
            
            # Then use Ollama to refine and improve the summary
            if self.ollama_client.is_available():
                prompt = f"""
                Please improve and refine this extractive summary by making it more coherent, 
                well-structured, and natural while preserving the key information:
                
                Extractive Summary: {extractive_summary}
                
                Original Text (for context): {text[:2000]}...
                
                Improved Summary:
                """
                
                return self.ollama_client.generate_summary(extractive_summary, prompt)
            else:
                # Fallback to extractive only
                return extractive_summary
                
        except Exception as e:
            logger.warning(f"Hybrid summarization failed, falling back to extractive: {e}")
            return self._extractive_summarize(text, sentence_count)
```

---

## 🎨 Phase 4: Streamlit UI Development

### Step 1: Main Application (app/main.py)
```python
"""
Main Streamlit application for Blog & PDF Summarizer.
"""
import streamlit as st
import logging
from pathlib import Path
import sys
import os

# Add app directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from app.config import Config
from app.utils.blog_scraper import BlogScraper
from app.utils.pdf_processor import PDFProcessor
from app.utils.text_summarizer import TextSummarizer
from app.utils.ollama_client import OllamaClient

# Initialize configuration and ensure directories exist
Config.ensure_directories()

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOG_DIR}/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Gen Z styling
def load_css():
    """Load custom CSS for Gen Z UI design."""
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Root variables */
    :root {
        --primary-bg: #0a0a0a;
        --secondary-bg: rgba(20, 20, 20, 0.8);
        --card-bg: rgba(30, 30, 30, 0.9);
        --text-primary: #ffffff;
        --text-secondary: #cccccc;
        --accent-cyan: #00FFFF;
        --accent-pink: #FF1493;
        --accent-purple: #8A2BE2;
        --gradient: linear-gradient(135deg, var(--accent-cyan), var(--accent-pink), var(--accent-purple));
    }
    
    /* Main app styling */
    .stApp {
        background: var(--primary-bg);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px var(--accent-cyan)); }
        to { filter: drop-shadow(0 0 20px var(--accent-pink)); }
    }
    
    /* Card styling */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 255, 255, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 5px 20px rgba(255, 20, 147, 0.4) !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: var(--secondary-bg) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--accent-cyan) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: var(--secondary-bg) !important;
        border: 2px dashed var(--accent-purple) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--card-bg) !important;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 15px !important;
        border: none !important;
        font-weight: 500 !important;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def check_ollama_status():
    """Check if Ollama is available and display status."""
    client = OllamaClient()
    if client.is_available():
        st.sidebar.success("🟢 Ollama Connected")
        models = client.get_available_models()
        if models:
            st.sidebar.info(f"Available models: {', '.join(models)}")
    else:
        st.sidebar.error("🔴 Ollama Disconnected")
        st.sidebar.warning("Please start Ollama service: `ollama serve`")

def main():
    """Main application function."""
    # Load custom CSS
    load_css()
    
    # App header
    st.markdown('<h1 class="main-header">AI Blog & PDF Summarizer 🤖</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #cccccc; font-size: 1.2rem; margin-bottom: 2rem;">Powered by Ollama - Summarize Smarter, Read Faster</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Check Ollama status
        check_ollama_status()
        
        # Summarization settings
        st.markdown("### 📝 Summarization")
        summary_method = st.selectbox(
            "Method",
            ["ollama", "extractive", "hybrid"],
            help="Choose summarization approach"
        )
        
        summary_length = st.selectbox(
            "Length",
            ["short", "medium", "long"],
            index=1,
            help="Summary detail level"
        )
        
        st.markdown("### 🎨 Theme")
        theme = st.selectbox("Choose Theme", ["Dark (Gen Z)", "Classic"], index=0)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🌐 Blog Summarizer")
        
        # Blog URL input
        blog_url = st.text_input(
            "Enter Blog URL",
            placeholder="https://example.com/article...",
            help="Paste any blog or article URL"
        )
        
        if st.button("📄 Summarize Blog", key="blog_btn"):
            if blog_url:
                summarize_blog(blog_url, summary_method, summary_length)
            else:
                st.error("Please enter a valid URL")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📁 PDF Summarizer")
        
        # PDF file upload
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=['pdf'],
            help="Upload PDF document (max 10MB)"
        )
        
        if uploaded_file and st.button("📊 Summarize PDF", key="pdf_btn"):
            summarize_pdf(uploaded_file, summary_method, summary_length)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results area
    if 'summary_result' in st.session_state:
        display_results(st.session_state.summary_result)

def summarize_blog(url: str, method: str, length: str):
    """Summarize blog content from URL."""
    try:
        with st.spinner("🔍 Extracting blog content..."):
            scraper = BlogScraper()
            content = scraper.extract_content(url)
        
        st.success(f"✅ Extracted {content['word_count']} words from: {content['title']}")
        
        if content['content']:
            with st.spinner("🤖 Generating summary..."):
                summarizer = TextSummarizer()
                result = summarizer.summarize(content['content'], method, length)
                result['source_type'] = 'blog'
                result['source_title'] = content['title']
                result['source_url'] = url
                st.session_state.summary_result = result
        else:
            st.error("No content could be extracted from the URL")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Blog summarization failed: {e}")

def summarize_pdf(uploaded_file, method: str, length: str):
    """Summarize PDF document."""
    try:
        with st.spinner("📖 Processing PDF..."):
            processor = PDFProcessor()
            content = processor.extract_text(uploaded_file, uploaded_file.name)
        
        st.success(f"✅ Extracted {content['word_count']} words from {content['page_count']} pages")
        
        if content['text']:
            with st.spinner("🤖 Generating summary..."):
                summarizer = TextSummarizer()
                result = summarizer.summarize(content['text'], method, length)
                result['source_type'] = 'pdf'
                result['source_title'] = content['title']
                result['source_filename'] = content['filename']
                st.session_state.summary_result = result
        else:
            st.error("No text could be extracted from the PDF")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"PDF summarization failed: {e}")

def display_results(result: dict):
    """Display summarization results."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Summary Results")
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Compression", f"{result['compression_ratio']}%")
    with col2:
        st.metric("Original Words", result['original_words'])
    with col3:
        st.metric("Summary Words", result['summary_words'])
    
    # Source info
    if result['source_type'] == 'blog':
        st.info(f"🌐 **Source:** [{result['source_title']}]({result['source_url']})")
    else:
        st.info(f"📄 **Source:** {result['source_filename']}")
    
    # Summary content
    st.markdown("#### Generated Summary")
    st.markdown(f'<div style="background: rgba(0, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; border-left: 4px solid #00FFFF;">{result["summary"]}</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy Summary", key="copy_btn"):
            st.code(result['summary'])
            st.success("Summary copied!")
    
    with col2:
        if st.button("🔄 New Summary", key="new_btn"):
            del st.session_state.summary_result
            st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
```

---

## 🚀 Phase 5: Running and Testing

### Step 1: Start Ollama Service
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Verify model availability
ollama list
```

### Step 2: Run the Application
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run Streamlit app
streamlit run app/main.py

# Or use VS Code debugger (F5)
```

### Step 3: Test the Application

1. **Open browser** to `http://localhost:8501`
2. **Test blog summarization**:
   - Enter a blog URL (e.g., a Medium article)
   - Click "Summarize Blog"
   - Verify content extraction and summarization
3. **Test PDF summarization**:
   - Upload a PDF document
   - Click "Summarize PDF"
   - Verify text extraction and summarization

---

## 🔧 Phase 6: Advanced Features & Optimization

### Step 1: Add Error Handling & Logging
Create comprehensive error handling for all user interactions and log important events for debugging.

### Step 2: Performance Optimization
- Implement caching for repeated URLs
- Add progress bars for long operations
- Optimize memory usage for large files

### Step 3: Security Enhancements
- Add input validation and sanitization
- Implement file type verification
- Add rate limiting for API calls

### Step 4: UI/UX Improvements
- Add dark/light theme toggle
- Implement responsive design
- Add keyboard shortcuts
- Include loading animations

---

## 📦 Phase 7: Deployment Options

### Option 1: Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Set environment variables
4. Deploy automatically

### Option 2: Docker Deployment
Create Dockerfile for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Option 3: Local Network Deployment
Run on local network for team access:
```bash
streamlit run app/main.py --server.address 0.0.0.0
```

---

## 🐛 Troubleshooting Guide

### Common Issues:

1. **Ollama Connection Failed**
   - Ensure Ollama is running: `ollama serve`
   - Check if model is pulled: `ollama list`
   - Verify port 11434 is available

2. **PDF Processing Errors**
   - Check file size limits
   - Verify PDF isn't password-protected
   - Ensure sufficient memory for large files

3. **Blog Scraping Issues**
   - Some sites block scrapers - try different User-Agent
   - Check for CAPTCHA or authentication requirements
   - Verify URL is accessible

4. **Import Errors**
   - Ensure virtual environment is activated
   - Check all dependencies are installed
   - Verify Python path configuration

---

## 🎯 Next Steps & Extensions

### Potential Enhancements:
1. **Multi-language support** for international blogs
2. **Batch processing** for multiple files
3. **Summary comparison** between different models
4. **Export options** (PDF, Word, etc.)
5. **User accounts** and history tracking
6. **API integration** with other services
7. **Mobile app** version
8. **Browser extension** for one-click summarization

This comprehensive guide provides everything needed to build a professional-grade blog and PDF summarizer with a modern Gen Z interface. The modular architecture makes it easy to extend and maintain while providing excellent performance and user experience.