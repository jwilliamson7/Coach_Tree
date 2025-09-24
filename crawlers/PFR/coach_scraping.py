import requests
import time
import os
import pandas as pd
from random import randint
from pathlib import Path
from bs4 import BeautifulSoup, Comment
from typing import List, Optional, Tuple


class CoachDataScraper:
    """Scrapes NFL coach data from pro-football-reference.com"""
    
    BASE_URL = 'https://www.pro-football-reference.com'
    
    # Table IDs and their corresponding data types
    TABLE_CONFIGS = {
        "all_coaching_results": {
            "name": "all_coaching_results",
            "duplicate_headers": ['Playoffs']
        },
        "all_coaching_ranks": {
            "name": "all_coaching_ranks", 
            "duplicate_headers": ['Rushing Off', 'Passing Off', 'Defense', 'Rushing Def', 'Passing Def']
        },
        "all_coaching_history": {
            "name": "all_coaching_history",
            "duplicate_headers": []
        }
    }
    
    def __init__(self, output_dir: str = "../../data/raw/Coaches"):
        """Initialize the scraper with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_coach_directory(self, coach_name: str) -> Path:
        """Create directory for coach data"""
        coach_dir = self.output_dir / coach_name
        coach_dir.mkdir(exist_ok=True)
        return coach_dir
        
    def _extract_table_from_div(self, table_div) -> Optional[BeautifulSoup]:
        """Extract table from div, checking both direct content and HTML comments"""
        if table_div is None:
            return None
            
        table_soup = BeautifulSoup(str(table_div), "lxml")
        table = table_soup.find('table')
        
        # If no direct table found, check HTML comments
        if table is None:
            comment = table_soup.find(text=lambda text: isinstance(text, Comment))
            if comment and comment.find("<table ") >= 0:
                comment_soup = BeautifulSoup(str(comment), 'html.parser')
                table = comment_soup.find("table")
                
        return table
        
    def _extract_headers(self, table: BeautifulSoup, duplicate_headers: List[str]) -> List[str]:
        """Extract and deduplicate table headers"""
        headers = []
        header_index = 0
        
        for th in table.find_all("th"):
            class_list = th.get('class', [])
            if "poptip" in class_list:
                header_text = th.text.strip()
                
                # Handle duplicate headers by appending suffix
                while header_text in headers:
                    if header_index < len(duplicate_headers):
                        header_text = f"{th.text.strip()} {duplicate_headers[header_index]}"
                        header_index += 1
                    else:
                        header_text = f"{th.text.strip()}_{len([h for h in headers if h.startswith(th.text.strip())])}"
                        
                headers.append(header_text)
                
        return headers
        
    def _extract_table_rows(self, table: BeautifulSoup) -> List[List[str]]:
        """Extract data rows from table"""
        rows = []
        tbody = table.find('tbody')
        
        if tbody:
            for tr in tbody.find_all('tr'):
                row = []
                
                # First cell is usually in th
                th = tr.find('th')
                if th:
                    row.append(th.text.strip())
                    
                # Remaining cells in td
                for td in tr.find_all('td'):
                    row.append(td.text.strip())
                    
                if row:  # Only add non-empty rows
                    rows.append(row)
                    
        return rows
        
    def _save_table_data(self, headers: List[str], rows: List[List[str]], 
                        table_name: str, coach_dir: Path) -> None:
        """Save table data to CSV file"""
        if headers and rows:
            df = pd.DataFrame(data=rows, columns=headers)
            csv_path = coach_dir / f"{table_name}.csv"
            df.to_csv(csv_path, index=False)
            
    def scrape_coach_data(self, coach_url: str) -> bool:
        """Scrape all coaching data for a single coach"""
        try:
            # Fetch coach page
            full_url = f"{self.BASE_URL}{coach_url}"
            response = requests.get(full_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract coach name from page title
            coach_name = soup.find('h1').text.strip()
            coach_dir = self._create_coach_directory(coach_name)
            
            # Process each table type
            for table_id, config in self.TABLE_CONFIGS.items():
                table_div = soup.find("div", {"id": table_id})
                table = self._extract_table_from_div(table_div)
                
                if table:
                    headers = self._extract_headers(table, config["duplicate_headers"])
                    rows = self._extract_table_rows(table)
                    self._save_table_data(headers, rows, config["name"], coach_dir)
                    
            return True
            
        except Exception as e:
            print(f"Error scraping coach data from {coach_url}: {e}")
            return False

    def _rate_limit(self, last_request_time: float, min_delay: float = 1.0) -> None:
        """Implement rate limiting between requests"""
        elapsed = time.time() - last_request_time
        if elapsed < min_delay:
            sleep_time = randint(1, 5)  # Random delay between 1-5 seconds
            time.sleep(sleep_time)
            
    def _discover_coach_urls(self) -> List[str]:
        """Discover all coach URLs from the main coaches page"""
        try:
            coaches_url = f"{self.BASE_URL}/coaches/"
            response = requests.get(coaches_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            coach_urls = []
            
            # Extract all coach profile links
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.startswith('/coaches/') and href.endswith('.htm'):
                    coach_urls.append(href)
                    
            return coach_urls
            
        except Exception as e:
            print(f"Error discovering coach URLs: {e}")
            return []
            
    def scrape_all_coaches(self, manual_urls: Optional[List[str]] = None) -> Tuple[int, int]:
        """Scrape data for all coaches with rate limiting and progress tracking"""
        
        # Default manual URLs for coaches that might not be easily discoverable
        if manual_urls is None:
            manual_urls = [
                '/coaches/JohnBe0.htm',
                '/coaches/CoenLi0.htm', 
                '/coaches/MoorKe0.htm',
                '/coaches/GlenAa0.htm',
                '/coaches/SchoBr0.htm'
            ]
        
        # Combine manual URLs with discovered URLs
        discovered_urls = self._discover_coach_urls()
        all_urls = manual_urls + [url for url in discovered_urls if url not in manual_urls]
        
        # Track progress and results
        successful_scrapes = 0
        failed_scrapes = 0
        last_request_time = 0
        
        # Create log file
        log_file = self.output_dir / 'visited.txt'
        
        with open(log_file, 'w') as log:
            for i, coach_url in enumerate(all_urls, 1):
                print(f"Scraping coach {i}/{len(all_urls)}: {coach_url}")
                
                # Rate limiting
                self._rate_limit(last_request_time)
                
                # Attempt to scrape coach data
                if self.scrape_coach_data(coach_url):
                    successful_scrapes += 1
                    log.write(f"{coach_url}\n")
                    log.flush()  # Ensure immediate write
                else:
                    failed_scrapes += 1
                    
                last_request_time = time.time()
                
        return successful_scrapes, failed_scrapes


def main():
    """Main execution function"""
    print("Starting NFL Coach Data Scraper...")
    
    # Initialize scraper
    scraper = CoachDataScraper()
    
    # Run the scraping process
    successful, failed = scraper.scrape_all_coaches()
    
    print(f"\nScraping completed!")
    print(f"Successfully scraped: {successful} coaches")
    print(f"Failed to scrape: {failed} coaches")
    print(f"Total attempts: {successful + failed}")


if __name__ == "__main__":
    main()