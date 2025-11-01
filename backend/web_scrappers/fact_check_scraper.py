import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import re

def scrape_factcheck():
    """
    Scrape fact-checks from FactCheck.org
    Based on actual site structure observed
    """
    base_url = "https://www.factcheck.org"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    data = []
    
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # FactCheck.org uses article items
        articles = soup.find_all('article', class_='item')
        if not articles:
            articles = soup.find_all('article')
        
        print(f"Found {len(articles)} articles on homepage")
        
        for article in articles[:12]:
            try:
                # Find article link
                link_tag = article.find('h3')
                if link_tag:
                    link = link_tag.find('a', href=True)
                else:
                    link = article.find('a', href=True)
                
                if not link:
                    continue
                
                article_url = link['href']
                if not article_url.startswith('http'):
                    article_url = base_url + article_url
                
                print(f"Fetching: {article_url}")
                
                # Fetch individual article
                article_response = requests.get(article_url, headers=headers, timeout=10)
                article_response.raise_for_status()
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                # Extract title
                title_tag = article_soup.find('h1', class_='entry-title')
                if not title_tag:
                    title_tag = article_soup.find('h1')
                title = title_tag.get_text(strip=True) if title_tag else ""
                
                # Extract article content
                content_div = article_soup.find('div', class_='entry-content')
                if not content_div:
                    content_div = article_soup.find('article')
                
                content_paragraphs = []
                if content_div:
                    paragraphs = content_div.find_all('p')
                    # Get first 5 meaningful paragraphs
                    for p in paragraphs[:8]:
                        text = p.get_text(strip=True)
                        if len(text) > 30:  # Skip short paragraphs
                            content_paragraphs.append(text)
                        if len(content_paragraphs) >= 5:
                            break
                
                content_text = ' '.join(content_paragraphs)
                full_content = f"Analysis: {title}. {content_text}"
                
                # Determine verdict based on content analysis
                verdict = "false"
                content_lower = content_text.lower()
                title_lower = title.lower()
                combined = f"{title_lower} {content_lower}"
                
                # FactCheck.org often states claims are false, misleading, or true
                true_indicators = ['is accurate', 'is correct', 'is true', 'that\'s accurate', 'that\'s correct', 'that\'s true', 'confirmed']
                false_indicators = ['no evidence', 'false', 'misleading', 'incorrect', 'not true', 'unfounded', 'unsubstantiated', 'offers no evidence']
                
                false_count = sum(1 for indicator in false_indicators if indicator in combined)
                true_count = sum(1 for indicator in true_indicators if indicator in combined)
                
                if false_count > true_count:
                    verdict = "false"
                elif true_count > false_count:
                    verdict = "true"
                
                # Extract category
                category_tag = article_soup.find('a', rel='category tag')
                category = "politics"  # FactCheck.org is primarily political
                if category_tag:
                    cat_text = category_tag.get_text(strip=True).lower()
                    if 'health' in cat_text or 'science' in cat_text:
                        category = "health"
                    elif 'science' in cat_text or 'tech' in cat_text:
                        category = "science"
                
                # Extract date
                date_tag = article_soup.find('time', class_='entry-date')
                if not date_tag:
                    date_tag = article_soup.find('time')
                
                date = datetime.now().strftime('%Y-%m-%d')
                if date_tag and date_tag.has_attr('datetime'):
                    date = date_tag['datetime'][:10]
                
                # Extract author
                author_tag = article_soup.find('a', rel='author')
                if not author_tag:
                    author_tag = article_soup.find('span', class_='author')
                
                author = author_tag.get_text(strip=True) if author_tag else "FactCheck.org Staff"
                
                if full_content and len(full_content) > 100:
                    data.append({
                        "content": full_content,
                        "metadata": {
                            "source_url": article_url,
                            "source_name": "FactCheck.org",
                            "verdict": verdict,
                            "category": category,
                            "date": date,
                            "author": author
                        }
                    })
                
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
    
    except Exception as e:
        print(f"Error fetching main page: {e}")
    
    return data

if __name__ == "__main__":
    print("Starting FactCheck.org scraper...")
    data = scrape_factcheck()
    
    if data:
        with open('factcheck_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\nScraped {len(data)} articles from FactCheck.org")
        print(f"Data saved to factcheck_data.json")
    else:
        print("\nNo data scraped. Please check if the site structure has changed.")