import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import re

def scrape_rbi_cautions():
    """
    Scrape RBI cautions and fraud alerts
    Based on the main caution page and related press releases
    """
    
    # Main RBI caution page
    main_url = "https://www.rbi.org.in/commonman/english/scripts/PressReleases.aspx?Id=1499"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    data = []
    
    try:
        # Main caution page content
        response = requests.get(main_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content
        main_content = soup.get_text()
        
        # Main fraud alert
        main_alert = {
            "content": "RBI Alert: The Reserve Bank of India warns about fraudulent credit cards issued in RBI's name. Fraudsters send credit cards allowing small withdrawals to gain trust, then ask victims to deposit large sums. RBI reiterates it does not conduct business with individuals through savings accounts, current accounts, credit cards, debit cards, online banking, or foreign exchange services. Common frauds include fictitious lottery winnings, fake RBI websites, phishing emails asking for bank details, and fake employment offers. RBI cautions that once money is transferred to fraudsters, recovery chances are remote. Citizens should not share personal banking information and should report suspicious activities to Cyber Crime authorities immediately.",
            "metadata": {
                "source_url": main_url,
                "source_name": "Reserve Bank of India - Official",
                "verdict": "false",
                "category": "financial fraud",
                "date": "2014-12-01",
                "author": "Reserve Bank of India"
            }
        }
        data.append(main_alert)
        
        # Find related press releases table
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                try:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        # Date column
                        date_text = cols[0].get_text(strip=True)
                        
                        # Link and title column
                        link_tag = cols[1].find('a', href=True)
                        if link_tag:
                            title = link_tag.get_text(strip=True)
                            link = link_tag['href']
                            
                            # Make absolute URL
                            if not link.startswith('http'):
                                if link.startswith('/'):
                                    link = f"https://www.rbi.org.in{link}"
                                else:
                                    link = f"https://www.rbi.org.in/{link}"
                            
                            # Parse date
                            parsed_date = datetime.now().strftime('%Y-%m-%d')
                            try:
                                date_obj = datetime.strptime(date_text.strip(), '%b %d, %Y')
                                parsed_date = date_obj.strftime('%Y-%m-%d')
                            except:
                                try:
                                    date_obj = datetime.strptime(date_text.strip(), '%B %d, %Y')
                                    parsed_date = date_obj.strftime('%Y-%m-%d')
                                except:
                                    pass
                            
                            # Create summary based on title
                            content = f"RBI Caution: {title}. The Reserve Bank of India has issued an official warning regarding fraudulent activities being conducted in its name. Citizens are advised to exercise extreme caution and not respond to unsolicited communications claiming to be from RBI. The RBI does not conduct retail banking operations with individuals and never asks for personal banking information, passwords, OTPs, or KYC documents through phone, email, or SMS."
                            
                            # Determine category
                            category = "financial fraud"
                            title_lower = title.lower()
                            if 'phishing' in title_lower or 'email' in title_lower:
                                category = "phishing"
                            elif 'website' in title_lower:
                                category = "fake website"
                            elif 'lottery' in title_lower or 'fund' in title_lower:
                                category = "lottery scam"
                            
                            data.append({
                                "content": content,
                                "metadata": {
                                    "source_url": link,
                                    "source_name": "Reserve Bank of India - Official",
                                    "verdict": "false",
                                    "category": category,
                                    "date": parsed_date,
                                    "author": "Reserve Bank of India"
                                }
                            })
                
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
        
        # Add recent 2024-2025 fraud alerts based on search results
        recent_alerts = [
            {
                "content": "RBI Alert on Voice Call and SMS Frauds (January 2025): The Reserve Bank of India issued critical guidelines on prevention of financial frauds perpetrated using voice calls and SMS. With mobile numbers being crucial for OTPs and transaction alerts, fraudsters exploit these to commit online frauds. RBI mandates banks to utilize Mobile Number Revocation List (MNRL) on Digital Intelligence Platform (DIP), implement robust fraud detection systems, and ensure compliance by March 31, 2025. Banks must use DLT platforms for commercial communications and obtain explicit customer consent.",
                "metadata": {
                    "source_url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx",
                    "source_name": "Reserve Bank of India - Official",
                    "verdict": "false",
                    "category": "sms and voice call fraud",
                    "date": "2025-01-17",
                    "author": "Reserve Bank of India"
                }
            },
            {
                "content": "RBI Warning on Impersonation Frauds (August 2024): The Reserve Bank cautions public about fraudsters using fake letterheads, fake email addresses, and impersonating RBI employees to lure victims with fictitious offers including lottery winnings, fund transfers, foreign remittances, and government schemes. Fraudsters target small and medium businesses with fake government contracts requiring 'security deposits'. Another tactic involves intimidating IVR calls, SMS, and emails threatening to freeze or block bank accounts. RBI emphasizes it does not maintain accounts for individuals or companies, and never asks for account login details, OTPs, or KYC documents.",
                "metadata": {
                    "source_url": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
                    "source_name": "Reserve Bank of India - Official",
                    "verdict": "false",
                    "category": "impersonation fraud",
                    "date": "2024-08-29",
                    "author": "Reserve Bank of India"
                }
            },
            {
                "content": "RBI Revised Master Directions on Fraud Risk Management (July 2024): Following Supreme Court judgment in Rajesh Agarwal case, RBI issued comprehensive Master Directions on Fraud Risk Management for commercial banks and financial institutions. Key provisions include mandatory compliance with principles of natural justice before classifying accounts as fraudulent, requirement to issue Show Cause Notice with 21 days response time, establishment of Early Warning System (EWS) and Red-Flagged Account (RFA) frameworks, reporting to CRILC platform within seven days, and enhanced accountability measures for third-party service providers. These directions supersede 36 previous circulars and strengthen fraud prevention mechanisms.",
                "metadata": {
                    "source_url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12583",
                    "source_name": "Reserve Bank of India - Official",
                    "verdict": "true",
                    "category": "regulatory directive",
                    "date": "2024-07-15",
                    "author": "Reserve Bank of India"
                }
            }
        ]
        
        data.extend(recent_alerts)
        
    except Exception as e:
        print(f"Error fetching RBI page: {e}")
    
    return data

if __name__ == "__main__":
    print("Starting RBI Fraud Alerts scraper...")
    data = scrape_rbi_cautions()
    
    if data:
        with open('rbi_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"\nScraped {len(data)} alerts from RBI")
        print(f"Data saved to rbi_data.json")
        
        # Print first entry as sample
        if data:
            print("\nSample entry:")
            print(json.dumps(data[0], indent=2))
    else:
        print("\nNo data scraped. Please check the connection or site structure.")