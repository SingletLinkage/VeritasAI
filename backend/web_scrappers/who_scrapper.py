import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re

def scrape_who():
    """
    Scrape COVID-19 myth busters from WHO
    Based on actual site structure: FACT: statements with explanations
    """
    url = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    data = []
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get all text content
        page_text = soup.get_text()
        
        # WHO Myth Busters format: "FACT: <statement>"
        # Split by FACT: to get individual myth busters
        fact_sections = re.split(r'FACT:\s*', page_text)
        
        print(f"Found {len(fact_sections)-1} FACT sections")
        
        for i, section in enumerate(fact_sections[1:], 1):  # Skip first empty split
            try:
                # Extract the fact statement and explanation
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                
                if not lines or len(lines) < 2:
                    continue
                
                # First line is usually the myth/fact statement
                fact_statement = lines[0]
                
                # Rest is explanation - take up to 5 lines
                explanation_lines = []
                for line in lines[1:8]:
                    # Skip lines that are just navigation or irrelevant
                    if len(line) > 20 and not line.startswith('WHAT YOU CAN DO'):
                        explanation_lines.append(line)
                    if len(explanation_lines) >= 5:
                        break
                
                explanation = ' '.join(explanation_lines)
                
                # Construct full content
                full_content = f"WHO Myth Buster - FACT: {fact_statement}. Explanation: {explanation}"
                
                # WHO myth busters debunk false claims, so most are clarifying truth
                # But the myths themselves are false
                verdict = "false"  # The myths being busted are false
                
                # However, if the fact is stating something affirmative
                fact_lower = fact_statement.lower()
                if any(phrase in fact_lower for phrase in [
                    'you can', 'it is safe', 'is effective', 'does protect',
                    'are safe', 'can be used', 'is listed'
                ]):
                    verdict = "true"
                
                # All WHO myth busters are health-related
                category = "health"
                
                # WHO doesn't always date individual myth busters
                # Use a reasonable date from when the page was created
                date = "2023-01-19"  # Date shown on the page
                
                author = "World Health Organization"
                
                if len(full_content) > 100:
                    data.append({
                        "content": full_content,
                        "metadata": {
                            "source_url": url,
                            "source_name": "WHO Myth Busters",
                            "verdict": verdict,
                            "category": category,
                            "date": date,
                            "author": author
                        }
                    })
                
            except Exception as e:
                print(f"Error processing section {i}: {e}")
                continue
        
        # Alternative approach: Look for specific HTML structures
        if len(data) < 5:
            print("Trying alternative parsing method...")
            
            # Try to find content blocks or sections
            content_blocks = soup.find_all(['div', 'section'], class_=re.compile(r'content|block|section'))
            
            for block in content_blocks:
                text = block.get_text()
                if 'FACT:' in text and len(text) > 100:
                    # Extract fact statement
                    fact_match = re.search(r'FACT:\s*([^\n]+)', text)
                    if fact_match:
                        fact_statement = fact_match.group(1).strip()
                        
                        # Get paragraphs after FACT
                        paragraphs = block.find_all('p')
                        explanation = ' '.join([p.get_text(strip=True) for p in paragraphs[:3]])
                        
                        full_content = f"WHO Myth Buster - FACT: {fact_statement}. Explanation: {explanation}"
                        
                        verdict = "false"
                        if any(phrase in fact_statement.lower() for phrase in [
                            'you can', 'it is safe', 'is effective', 'does protect'
                        ]):
                            verdict = "true"
                        
                        if len(full_content) > 100:
                            data.append({
                                "content": full_content,
                                "metadata": {
                                    "source_url": url,
                                    "source_name": "WHO Myth Busters",
                                    "verdict": verdict,
                                    "category": "health",
                                    "date": "2023-01-19",
                                    "author": "World Health Organization"
                                }
                            })
    
    except Exception as e:
        print(f"Error fetching WHO page: {e}")
    
    return data

if __name__ == "__main__":
    print("Starting WHO Myth Busters scraper...")
    data = scrape_who()
    
    if data:
        with open('who_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\nScraped {len(data)} myth busters from WHO")
        print(f"Data saved to who_data.json")
        
        # Print first entry as sample
        if data:
            print("\nSample entry:")
            print(json.dumps(data[0], indent=2))
    else:
        print("\nNo data scraped. Please check if the site structure has changed.")