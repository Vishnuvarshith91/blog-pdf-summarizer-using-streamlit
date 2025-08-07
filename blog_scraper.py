"""
Utility to scrape and extract blog content for summarization.
"""
import requests
from bs4 import BeautifulSoup
import validators
import logging

logger = logging.getLogger(__name__)

class BlogScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Bot/1.0; +https://example.com/bot)'
        })

    def extract_content(self, url: str) -> dict:
        if not validators.url(url):
            raise ValueError("Invalid URL provided.")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            title = self._extract_title(soup)
            content = self._extract_main_content(soup)
            description = self._extract_description(soup)

            return {
                'title': title,
                'content': content,
                'description': description,
                'url': url,
                'word_count': len(content.split()) if content else 0
            }

        except requests.RequestException as e:
            logger.error(f"Request error scraping {url}: {e}")
            raise RuntimeError(f"Error fetching blog content: {e}")
        except Exception as e:
            logger.error(f"Error extracting blog content from {url}: {e}")
            raise RuntimeError(f"Error processing blog content: {e}")

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for tag in ['h1', 'title', '[property="og:title"]']:
            element = soup.select_one(tag)
            if element and element.get_text(strip=True):
                return element.get_text(strip=True)
        return "No Title Found"

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        # Remove unwanted tags
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        # Try common containers
        candidates = ['article', '.post-content', '.entry-content', '.content', 'main']
        for selector in candidates:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 100:
                    return text
        # Fallback to concatenated paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        return text

    def _extract_description(self, soup: BeautifulSoup) -> str:
        for meta in ['meta[property="og:description"]', 'meta[name="description"]']:
            element = soup.select_one(meta)
            if element and element.has_attr('content'):
                return element['content']
        return ""
