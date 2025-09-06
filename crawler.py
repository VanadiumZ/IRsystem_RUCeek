import hashlib
import os
import random
import time
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup
from url_normalize import url_normalize

from filter import val_url

def get_html(uri, headers=None, timeout=None):
    """Fetch HTML content from a web page
    
    Args:
        uri (str): Target URL
        headers (dict, optional): HTTP request headers
        timeout (int, optional): Request timeout in seconds
        default_encoding (str, optional): Default encoding
        
    Returns:
        str: HTML content, returns "000000" on failure
    """
    if headers is None:
        headers = {}
        
    try:
        response = requests.get(uri, headers=headers, timeout=timeout)
        print(f"{uri} - Status: {response.status_code}")
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        print(f"Failed to fetch {uri}: {e}")
        return "000000"

def crawl_all_urls(html_doc, base_url):
    """Extract all links from an HTML document
    
    Args:
        html_doc (str): HTML document content
        base_url (str): Base URL for converting relative links
        
    Returns:
        set: Set of normalized URLs
    """
    all_links = set()
    try:
        soup = BeautifulSoup(html_doc, 'html.parser')
    except Exception as e:
        print(f"Failed to parse HTML document: {e}")
        return all_links
    
    for anchor in soup.find_all('a'):
        href = anchor.attrs.get('href')
        if not href:
            continue
        href = href.strip()
        if href.startswith(("javascript:", "mailto:", "tel:")):
            print("Skipping non-http link:", href)
        abs_url = urljoin(base_url, href)
        # delete fragment (#xxx)
        abs_url, _ = urldefrag(abs_url)
        if href and href.strip():  
            if not href.startswith('http'):
                href = urljoin(base_url, href)
            all_links.add(url_normalize(href))
    return all_links

def get_file_path(url, count):
    """Generate file path based on URL and count
    
    Args:
        url (str): Web page URL
        count (int): File counter
        
    Returns:
        str: Generated filename
    """
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return f"{count}_{url_hash}.html"

def main():
    """Main function: Execute web crawling task"""
    # Configuration parameters
    # Manually set seed URLs, headers, target domains, and output directory
    seed_urls = ['http://xsc.ruc.edu.cn/']
    headers = {'user-agent': 'MyCrawler/2.0 (Windows NT 10.0; Win64; x64; en-US)'}
    target_domains = ["keyan.ruc.edu.cn", "xsc.ruc.edu.cn"]
    output_dir = "tempHTML"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize crawler state
    url_queue = []
    all_urls = set()
    visited_urls = set()
    # set initial file count to avoid overwriting existing files (acording to previous runs)
    file_count = 3032
    
    # Add seed URLs to queue
    for url in seed_urls:
        if url not in all_urls:
            url_queue.append(url)
            all_urls.add(url)
    
    print(f"Starting crawl, seed URLs count: {len(seed_urls)}")
    print(f"Target domains: {target_domains}")
    
    # Main crawling loop
    while url_queue:
        current_url = url_queue.pop(0)
        visited_urls.add(current_url)
        
        print(f"Processing: {current_url}")
        
        # Fetch web page content
        html_content = get_html(current_url, headers=headers)
        if html_content == "000000":
            continue
            
        # Extract all links from the page
        discovered_urls = crawl_all_urls(html_content, current_url)
        
        # Add newly discovered valid links to queue
        new_urls_count = 0
        for new_url in discovered_urls:
            if new_url not in all_urls and val_url(new_url, target_domains):
                url_queue.append(new_url)
                all_urls.add(new_url)
                new_urls_count += 1
        
        print(f"New links found: {new_urls_count}, Queue remaining: {len(url_queue)}")
        
        # Save qualified web pages
        if val_url(current_url, target_domains):
            file_count += 1
            file_path = os.path.join(output_dir, get_file_path(current_url, file_count))
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"<!-- URL: {current_url} -->\n")
                    f.write(html_content)
                print(f"File saved {file_count}: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to save file: {e}")
        
        # Random delay to avoid overwhelming the server
        wait_time = random.uniform(0.1, 0.5)
        time.sleep(wait_time)
    
    print(f"\nCrawling completed!")
    print(f"Total visited: {len(visited_urls)} URLs")
    print(f"Links discovered: {len(all_urls)}")
    print(f"Files saved: {file_count - 5528}")


if __name__ == "__main__":
    main()






