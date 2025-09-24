import requests
import time
import os
import pandas as pd
from random import randint
from pathlib import Path
from bs4 import BeautifulSoup, Comment
import math
from typing import List, Optional, Tuple


class TeamDataScraper:
    """Scrapes NFL team data from pro-football-reference.com"""
    
    BASE_URL = "https://www.pro-football-reference.com/teams/"
    
    # Standard team statistics headers for yearly data
    YEARLY_STATS_HEADERS = [
        'Year', 'PF (Points For)', 'Yds', 'Offensive Plays', 'Y/P', 'TO', 'FL+', '1stD',
        'Cmp Passing', 'Att Passing', 'Yds Passing', 'TD Passing', 'Int Passing', 
        'NY/A Passing', '1stD Passing', 'Att Rushing', 'Yds Rushing', 'TD Rushing',
        'Y/A Rushing', '1stD Rushing', 'Pen', 'Yds Penalties', '1stPy', '#Dr', 'Sc%',
        'TO%', 'Start Average Drive', 'Time Average Drive', 'Plays Average Drive',
        'Yds Average Drive', 'Pts Average Drive', '3DAtt', '3DConv', '3D%', '4DAtt',
        '4DConv', '4D%', 'RZAtt', 'RZTD', 'RZPct'
    ]
    
    def __init__(self, output_dir: str = "../../data/raw/Teams"):
        """Initialize the scraper with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_team_directory(self, team_abbrev: str) -> Path:
        """Create directory for team data"""
        team_dir = self.output_dir / team_abbrev
        team_dir.mkdir(exist_ok=True)
        return team_dir
        
    def _rate_limit(self, start_time: float, threshold_seconds: float = 5.0) -> None:
        """Implement intelligent rate limiting"""
        elapsed = time.time() - start_time
        if elapsed < threshold_seconds:
            pause_duration = randint(0, 1) + math.ceil(threshold_seconds - elapsed)
            print(f'***Waiting {pause_duration} seconds for rate limiting')
            time.sleep(pause_duration)
            
    def _extract_headers_with_duplicates(self, table: BeautifulSoup, 
                                       duplicate_suffixes: List[str]) -> List[str]:
        """Extract table headers and handle duplicates with suffixes"""
        headers = []
        suffix_index = 0
        
        for th in table.find_all("th"):
            class_list = th.get('class', [])
            if "poptip" in class_list:
                header_text = th.text.strip()
                
                # Handle duplicate headers
                while header_text in headers:
                    if suffix_index < len(duplicate_suffixes):
                        header_text = f"{th.text.strip()} {duplicate_suffixes[suffix_index]}"
                        suffix_index += 1
                    else:
                        # Fallback for unexpected duplicates
                        header_text = f"{th.text.strip()}_{suffix_index}"
                        suffix_index += 1
                        
                headers.append(header_text)
                
        return headers
        
    def _extract_table_rows(self, table: BeautifulSoup, skip_thead: bool = True) -> List[List[str]]:
        """Extract data rows from table, optionally skipping thead-classed rows"""
        rows = []
        tbody = table.find('tbody')
        
        if tbody:
            for tr in tbody.find_all('tr'):
                # Skip header rows if requested
                if skip_thead and tr.get('class') and "thead" in tr.get('class'):
                    continue
                    
                row = []
                
                # First cell (usually year or identifier)
                th = tr.find('th')
                if th:
                    row.append(th.text.strip())
                    
                # Data cells
                for td in tr.find_all('td'):
                    row.append(td.text.strip())
                    
                if row:
                    rows.append(row)
                    
        return rows
        
    def _scrape_team_record(self, team_abbrev: str, team_dir: Path) -> List[str]:
        """Scrape team record data and return list of years"""
        response = requests.get(f"{self.BASE_URL}{team_abbrev}")
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.find('div', {'id': 'div_team_index'}).find('table')
        
        headers = self._extract_headers_with_duplicates(table, ['Def Rank'])
        rows = self._extract_table_rows(table)
        
        # Save team record data
        df = pd.DataFrame(data=rows, columns=headers)
        df.to_csv(team_dir / "team_record.csv", index=False)
        
        # Extract years for further processing
        years = [row[0] for row in rows if row]
        return years

    def _scrape_yearly_stats(self, team_abbrev: str, years: List[str], 
                           team_dir: Path) -> None:
        """Scrape yearly team and opponent statistics"""
        team_rows = []
        opponent_rows = []
        start_time = time.time()
        
        for year in years:
            print(f"Scraping year {year}")
            
            # Rate limiting
            self._rate_limit(start_time)
            
            # Fetch year-specific page
            response = requests.get(f"{self.BASE_URL}{team_abbrev}/{year}.htm")
            response.raise_for_status()
            start_time = time.time()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Get main team stats table
            stats_table = soup.find('div', {'id': 'div_team_stats'})
            if not stats_table:
                continue
            stats_table = stats_table.find('table')
            
            # Get conversions table (only available after 1998)
            conversions_table = None
            if int(year) > 1998:
                conv_div = soup.find('div', {'id': 'div_team_conversions'})
                if conv_div:
                    conversions_table = conv_div.find('table')
            
            # Extract team and opponent rows (first two rows)
            tbody = stats_table.find('tbody')
            if tbody:
                data_rows = tbody.find_all('tr')[:2]  # Team and opponent data
                
                for row_idx, tr in enumerate(data_rows):
                    row = [year]
                    
                    # Extract data from main stats table
                    for td in tr.find_all('td'):
                        row.append(td.text.strip())
                    
                    # Extract data from conversions table if available
                    if conversions_table:
                        conv_tbody = conversions_table.find('tbody')
                        if conv_tbody:
                            conv_rows = conv_tbody.find_all('tr')
                            if row_idx < len(conv_rows):
                                for td in conv_rows[row_idx].find_all('td'):
                                    row.append(td.text.strip())
                    else:
                        # Pad with empty values if no conversions data
                        missing_cols = len(self.YEARLY_STATS_HEADERS) - len(row)
                        row.extend([''] * missing_cols)
                    
                    # Assign to team or opponent based on row index
                    if row_idx == 0:
                        team_rows.append(row)
                    else:
                        opponent_rows.append(row)
        
        # Save the data
        team_df = pd.DataFrame(data=team_rows, columns=self.YEARLY_STATS_HEADERS)
        team_df.to_csv(team_dir / "yearly_team_stats.csv", index=False)
        
        opponent_df = pd.DataFrame(data=opponent_rows, columns=self.YEARLY_STATS_HEADERS)
        opponent_df.to_csv(team_dir / "yearly_opponent_stats.csv", index=False)
        
    def _scrape_playoff_data(self, team_abbrev: str, team_dir: Path) -> None:
        """Scrape team playoff game data"""
        try:
            response = requests.get(f"{self.BASE_URL}{team_abbrev}/playoffs.htm")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find playoff game log table
            playoff_div = soup.find('div', {'id': 'div_playoff_game_log'})
            if not playoff_div:
                return
                
            table = playoff_div.find('table')
            if not table:
                return
            
            # Extract headers
            headers = ['Year'] + self._extract_headers_with_duplicates(table, ['Offense', 'Defense'])
            
            # Extract playoff game data with year tracking
            rows = []
            current_year = ''
            
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    # Check if this is a year header row
                    if tr.get('class') and "thead" in tr.get('class'):
                        current_year = tr.text.strip().split()[0]
                        continue
                    
                    # Extract game data
                    row = [current_year]
                    
                    th = tr.find('th')
                    if th:
                        row.append(th.text.strip())
                    
                    for td in tr.find_all('td'):
                        row.append(td.text.strip())
                    
                    if len(row) > 1:  # Only add rows with actual data
                        rows.append(row)
            
            # Save playoff data
            if rows:
                df = pd.DataFrame(data=rows, columns=headers)
                df.to_csv(team_dir / "team_playoff_record.csv", index=False)
                
        except Exception as e:
            print(f"Error scraping playoff data for {team_abbrev}: {e}")
            
    def scrape_team_data(self, team_abbrev: str) -> bool:
        """Scrape all data for a single team"""
        try:
            team_dir = self._create_team_directory(team_abbrev)
            
            # Scrape main team record and get years
            years = self._scrape_team_record(team_abbrev, team_dir)
            
            # Scrape yearly statistics
            self._scrape_yearly_stats(team_abbrev, years, team_dir)
            
            # Scrape playoff data
            self._scrape_playoff_data(team_abbrev, team_dir)
            
            return True
            
        except Exception as e:
            print(f"Error scraping team data for {team_abbrev}: {e}")
            return False

    def _discover_active_teams(self) -> List[str]:
        """Discover active team abbreviations from main teams page"""
        try:
            response = requests.get(f"{self.BASE_URL}")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            team_abbrevs = []
            
            # Find active teams table
            active_table = soup.find('table', {"id": "teams_active"})
            if active_table:
                for link in active_table.find_all('a'):
                    href = link.get('href', '')
                    if 'teams' in href:
                        # Extract team abbreviation from URL path
                        team_abbrev = href.split('/')[-2]
                        if team_abbrev:
                            team_abbrevs.append(team_abbrev)
            
            return team_abbrevs
            
        except Exception as e:
            print(f"Error discovering active teams: {e}")
            return []
            
    def _discover_inactive_teams(self) -> List[str]:
        """Discover inactive team abbreviations from HTML comments"""
        try:
            response = requests.get(f"{self.BASE_URL}")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            team_abbrevs = []
            
            # Find inactive teams in HTML comment
            inactive_div = soup.find('div', {"id": "all_teams_inactive"})
            if inactive_div:
                comment = inactive_div.find(text=lambda text: isinstance(text, Comment))
                if comment and comment.find("<table ") >= 0:
                    comment_soup = BeautifulSoup(str(comment), 'html.parser')
                    table = comment_soup.find("table")
                    
                    if table:
                        for link in table.find_all('a'):
                            href = link.get('href', '')
                            if 'teams' in href:
                                team_abbrev = href.split('/')[-2]
                                if team_abbrev:
                                    team_abbrevs.append(team_abbrev)
            
            return team_abbrevs
            
        except Exception as e:
            print(f"Error discovering inactive teams: {e}")
            return []
            
    def scrape_all_teams(self, include_inactive: bool = False) -> Tuple[int, int]:
        """Scrape data for all teams with progress tracking"""
        
        # Discover team abbreviations
        active_teams = self._discover_active_teams()
        all_teams = active_teams.copy()
        
        if include_inactive:
            inactive_teams = self._discover_inactive_teams()
            all_teams.extend(inactive_teams)
        
        print(f"Found {len(active_teams)} active teams")
        if include_inactive:
            print(f"Found {len(inactive_teams)} inactive teams")
        print(f"Total teams to scrape: {len(all_teams)}")
        
        # Track progress and results
        successful_scrapes = 0
        failed_scrapes = 0
        start_time = time.time()
        
        # Create log file
        log_file = self.output_dir / 'visited_teams.txt'
        
        with open(log_file, 'w') as log:
            for i, team_abbrev in enumerate(all_teams, 1):
                print(f"Scraping team {i}/{len(all_teams)}: {team_abbrev}")
                
                # Rate limiting
                self._rate_limit(start_time)
                
                # Attempt to scrape team data
                if self.scrape_team_data(team_abbrev):
                    successful_scrapes += 1
                    log.write(f"{team_abbrev}\n")
                    log.flush()
                else:
                    failed_scrapes += 1
                    
                start_time = time.time()
                
        return successful_scrapes, failed_scrapes


def main():
    """Main execution function"""
    print("Starting NFL Team Data Scraper...")
    
    # Initialize scraper
    scraper = TeamDataScraper()
    
    # Run the scraping process (active teams only by default)
    successful, failed = scraper.scrape_all_teams(include_inactive=False)
    
    print(f"\nScraping completed!")
    print(f"Successfully scraped: {successful} teams")
    print(f"Failed to scrape: {failed} teams")
    print(f"Total attempts: {successful + failed}")


if __name__ == "__main__":
    main()