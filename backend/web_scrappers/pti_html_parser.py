from bs4 import BeautifulSoup
import json
import re
from datetime import datetime

def parse_pti_html(html_content):
    """
    Parse PTI Fact Check HTML content directly
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data = []
    
    # Find all article containers
    articles = soup.find_all('div', class_='col-md-12')
    
    for article in articles:
        try:
            # Find the link to the article
            link_tag = article.find('a', href=True, class_='text-dark text-decoration-none')
            if not link_tag:
                continue
            
            article_url = link_tag['href']
            if not article_url.startswith('http'):
                article_url = f"https://www.ptinews.com{article_url}"
            
            # Extract title (inside h6 with span)
            title_tag = article.find('h6')
            title = ""
            if title_tag:
                span_tag = title_tag.find('span', class_='head')
                if span_tag:
                    # Remove the "PTI Fact Check:" prefix
                    title_text = span_tag.get_text(strip=True)
                    title = re.sub(r'PTI Fact Check:\s*', '', title_text)
            
            # Extract description/content
            desc_tag = article.find('p', class_='desc')
            description = ""
            if desc_tag:
                description = desc_tag.get_text(strip=True)
            
            # Combine title and description for content
            full_content = f"Fact-check: {title}. {description}"
            
            # Extract date
            date_tag = article.find('p', class_='r-time')
            date = datetime.now().strftime('%Y-%m-%d')
            if date_tag:
                date_text = date_tag.get_text(strip=True)
                try:
                    # Parse format: "Friday, Oct 31, 2025 19:10:46"
                    date_parts = date_text.split(',')
                    if len(date_parts) >= 2:
                        date_str = date_parts[1].strip().split()[0:3]  # Get "Oct 31 2025"
                        date_obj = datetime.strptime(' '.join(date_str), '%b %d %Y')
                        date = date_obj.strftime('%Y-%m-%d')
                except Exception as e:
                    date = date_text
                    print(f"Error parsing date: {date_text}, {e}")
            
            # Determine verdict from title/content
            verdict = "false"
            combined = f"{title.lower()} {description.lower()}"
            
            # PTI typically debunks false claims
            if any(word in combined for word in ['falsely', 'false', 'fake', 'misleading', 'unrelated', 'old', 'digitally altered', 'ai-generated', 'scripted']):
                verdict = "false"
            elif any(word in combined for word in ['verified', 'true', 'correct', 'confirmed']):
                verdict = "true"
            
            # Determine category
            category = "general"
            if any(word in combined for word in ['health', 'covid', 'vaccine', 'medical', 'hospital']):
                category = "health"
            elif any(word in combined for word in ['politics', 'election', 'minister', 'government', 'pm modi', 'bjp', 'congress']):
                category = "politics"
            elif any(word in combined for word in ['cricketer', 'sports', 'match', 'player']):
                category = "sports"
            
            # Extract author from description
            author = "PTI Fact Check Team"
            author_match = re.search(r'\((.*?),\s*PTI Fact Check\)', description)
            if author_match:
                author = author_match.group(1).strip()
            
            if full_content and len(full_content) > 100:
                data.append({
                    "content": full_content,
                    "metadata": {
                        "source_url": article_url,
                        "source_name": "PTI News Fact Check",
                        "verdict": verdict,
                        "category": category,
                        "date": date,
                        "author": author
                    }
                })
        
        except Exception as e:
            print(f"Error processing article: {e}")
            continue
    
    return data

# Sample HTML content (paste your HTML here)
html_content = """
[PASTE YOUR HTML CONTENT HERE]
"""

if __name__ == "__main__":
    # For testing, you can paste HTML content directly
    # Or read from a file
    
    # Option 1: Read from file
    try:
        with open('pti_fact_check.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print("Please save the HTML content to 'pti_fact_check.html' or paste it in the script")
        print("\nAlternatively, you can use the parse_pti_html() function directly with HTML string")
        html_content = None
    
    if html_content:
        data = parse_pti_html(html_content)
        
        if data:
            with open('pti_data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"\nParsed {len(data)} articles from PTI Fact Check")
            print(f"Data saved to pti_data.json")
            
            # Print first entry as sample
            if data:
                print("\nSample entry:")
                print(json.dumps(data[0], indent=2))
        else:
            print("\nNo data parsed. Please check the HTML structure.")