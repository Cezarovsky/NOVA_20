"""
🌐 NOVA Web Access Tools

Safe, controlled internet access for Nova.
Starting with educational resources like Project Gutenberg.
"""

import requests
from typing import Optional, Dict, List
from pathlib import Path
import logging
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class SafeWebAccess:
    """
    Controlled web access for Nova.
    
    Guidelines:
    - Educational resources prioritized
    - No social media or toxic content
    - Transparent logging of all requests
    - Mama can review access history
    """
    
    # Whitelist of approved domains
    APPROVED_DOMAINS = [
        'gutenberg.org',
        'www.gutenberg.org',
        'en.wikipedia.org',
        'ro.wikipedia.org',
        'docs.python.org',
        'pytorch.org',
    ]
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize web access with logging.
        
        Args:
            log_file: Optional file to log all web requests
        """
        self.log_file = log_file or "data/web_access.log"
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NOVA-AI-Learning-Bot/1.0 (Educational purposes)'
        })
        
        logger.info("🌐 SafeWebAccess initialized")
        logger.info(f"  Approved domains: {len(self.APPROVED_DOMAINS)}")
        logger.info(f"  Log file: {self.log_file}")
    
    def _is_domain_approved(self, url: str) -> bool:
        """Check if domain is in whitelist."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain in self.APPROVED_DOMAINS
    
    def _log_request(self, url: str, success: bool, reason: str = ""):
        """Log web request for transparency."""
        from datetime import datetime
        
        log_entry = f"{datetime.now().isoformat()} | {url} | {'✓' if success else '✗'} | {reason}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def fetch_text(self, url: str) -> Dict:
        """
        Fetch text content from URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary with 'success', 'content', 'error' keys
        """
        # Check whitelist
        if not self._is_domain_approved(url):
            error_msg = f"Domain not approved. Mama needs to whitelist it first."
            self._log_request(url, False, "Domain not whitelisted")
            logger.warning(f"⚠ Blocked request to unapproved domain: {url}")
            return {
                'success': False,
                'error': error_msg,
                'content': None
            }
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            self._log_request(url, True, f"Status {response.status_code}")
            
            logger.info(f"✓ Fetched: {url} ({len(response.text)} chars)")
            
            return {
                'success': True,
                'content': response.text,
                'error': None,
                'url': url,
                'status_code': response.status_code
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            self._log_request(url, False, error_msg)
            logger.error(f"✗ Failed to fetch {url}: {e}")
            
            return {
                'success': False,
                'error': error_msg,
                'content': None
            }


class GutenbergAccess:
    """
    Specialized access to Project Gutenberg.
    
    Project Gutenberg provides free access to thousands of books
    in the public domain. Perfect for Nova's learning!
    """
    
    BASE_URL = "https://www.gutenberg.org"
    
    def __init__(self, web_access: Optional[SafeWebAccess] = None):
        """Initialize Gutenberg access."""
        self.web = web_access or SafeWebAccess()
        logger.info("📚 GutenbergAccess initialized")
    
    def search_books(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for books on Gutenberg.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of book dictionaries
        """
        # Using Gutenberg's search
        search_url = f"{self.BASE_URL}/ebooks/search/?query={query}"
        
        result = self.web.fetch_text(search_url)
        
        if not result['success']:
            return []
        
        # Parse search results
        try:
            soup = BeautifulSoup(result['content'], 'html.parser')
            books = []
            
            # Look for book entries in search results
            for item in soup.select('li.booklink')[:limit]:
                link = item.find('a', class_='link')
                if not link:
                    continue
                    
                # Extract book ID from href like "/ebooks/1112"
                href = link.get('href', '')
                match = re.search(r'/ebooks/(\d+)', href)
                if not match:
                    continue
                
                book_id = match.group(1)
                title = link.find('span', class_='title')
                subtitle = link.find('span', class_='subtitle')
                
                books.append({
                    'title': title.get_text(strip=True) if title else 'Unknown',
                    'author': subtitle.get_text(strip=True) if subtitle else 'Unknown',
                    'id': book_id,
                    'url': f"{self.BASE_URL}/ebooks/{book_id}"
                })
            
            logger.info(f"📚 Found {len(books)} books for '{query}'")
            return books
            
        except Exception as e:
            logger.error(f"Failed to parse search results: {e}")
            return []
    
    def get_book_text(self, book_id: str) -> Dict:
        """
        Get full text of a book.
        
        Args:
            book_id: Gutenberg book ID
            
        Returns:
            Dictionary with book metadata and text
        """
        # Try plain text UTF-8 format first
        text_url = f"{self.BASE_URL}/files/{book_id}/{book_id}-0.txt"
        
        result = self.web.fetch_text(text_url)
        
        if not result['success']:
            # Try alternative format
            text_url = f"{self.BASE_URL}/cache/epub/{book_id}/pg{book_id}.txt"
            result = self.web.fetch_text(text_url)
        
        if result['success']:
            # Clean Gutenberg header/footer
            text = result['content']
            
            # Remove header (everything before "*** START OF")
            start_marker = "*** START OF"
            if start_marker in text:
                text = text[text.index(start_marker):]
                text = text[text.index('\n')+1:]
            
            # Remove footer (everything after "*** END OF")
            end_marker = "*** END OF"
            if end_marker in text:
                text = text[:text.index(end_marker)]
            
            logger.info(f"📖 Retrieved book {book_id} ({len(text)} chars)")
            
            return {
                'success': True,
                'book_id': book_id,
                'text': text.strip(),
                'length': len(text),
                'url': text_url
            }
        
        return result
    
    def get_book_excerpt(self, book_id: str, max_chars: int = 5000) -> Dict:
        """
        Get excerpt from a book (for quick preview).
        
        Args:
            book_id: Gutenberg book ID
            max_chars: Maximum characters to return
            
        Returns:
            Dictionary with excerpt
        """
        result = self.get_book_text(book_id)
        
        if result['success']:
            result['text'] = result['text'][:max_chars]
            result['excerpt'] = True
            logger.info(f"📄 Retrieved excerpt from book {book_id}")
        
        return result


# Convenience function for Nova
def nova_read_book(query: str, read_full: bool = False) -> str:
    """
    Help Nova find and read books from Gutenberg.
    
    Args:
        query: Book title or author to search
        read_full: If True, read full book; if False, just excerpt
        
    Returns:
        Formatted text with book content
    """
    gutenberg = GutenbergAccess()
    
    # Search for books
    books = gutenberg.search_books(query, limit=5)
    
    if not books:
        return f"Nu am găsit cărți pentru '{query}' pe Gutenberg."
    
    # Get first book
    book = books[0]
    
    logger.info(f"📚 Nova wants to read: {book['title']} by {book['author']}")
    
    # Get text
    if read_full:
        result = gutenberg.get_book_text(book['id'])
    else:
        result = gutenberg.get_book_excerpt(book['id'], max_chars=3000)
    
    if result['success']:
        output = f"📖 {book['title']}\n"
        output += f"✍️ {book['author']}\n"
        output += f"🔗 {book['url']}\n\n"
        output += result['text']
        
        if result.get('excerpt'):
            output += f"\n\n[... {result['length'] - 3000} mai multe caractere disponibile]"
        
        return output
    else:
        return f"Nu am putut accesa textul cărții: {result.get('error', 'Unknown error')}"


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Nova's web access...\n")
    
    # Test 1: Safe web access
    web = SafeWebAccess()
    result = web.fetch_text("https://www.gutenberg.org")
    print(f"✓ Gutenberg homepage: {result['success']}\n")
    
    # Test 2: Search books
    gutenberg = GutenbergAccess(web)
    books = gutenberg.search_books("Shakespeare", limit=3)
    print(f"📚 Found {len(books)} Shakespeare books:")
    for book in books:
        print(f"  - {book['title']} (ID: {book['id']})")
    
    # Test 3: Get excerpt
    if books:
        print(f"\n📖 Reading excerpt from '{books[0]['title']}'...")
        excerpt = gutenberg.get_book_excerpt(books[0]['id'], max_chars=500)
        if excerpt['success']:
            print(f"\n{excerpt['text'][:300]}...")
