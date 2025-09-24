#!/usr/bin/env python3
"""
NFL Play-by-Play Data Downloader

Downloads NFL play-by-play data using nfl_data_py library for specified years.
Data is available from 1999 onwards.

Usage:
    python download_play_by_play.py --year 1999
    python download_play_by_play.py --start_year 1999 --end_year 2001
    python download_play_by_play.py --years 1999,2000,2001
"""

import argparse
import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
import sys
import logging
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlayByPlayDownloader:
    """Downloads and saves NFL play-by-play data"""
    
    def __init__(self, output_dir: str = "data/raw/play_by_play"):
        """Initialize downloader with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data availability starts in 1999
        self.min_year = 1999
        self.max_year = 2024  # Current season
        
    def validate_years(self, years: List[int]) -> List[int]:
        """Validate that years are within available range"""
        valid_years = []
        
        for year in years:
            if year < self.min_year:
                logger.warning(f"Year {year} is before data availability (starts in {self.min_year}). Skipping.")
            elif year > self.max_year:
                logger.warning(f"Year {year} is beyond current season ({self.max_year}). Skipping.")
            else:
                valid_years.append(year)
        
        return sorted(valid_years)
    
    def download_season(self, year: int) -> bool:
        """Download play-by-play data for a single season"""
        try:
            logger.info(f"Downloading play-by-play data for {year} season...")
            
            # Download data using nfl_data_py
            pbp_data = nfl.import_pbp_data([year])
            
            if pbp_data.empty:
                logger.warning(f"No data returned for {year}")
                return False
            
            # Save to CSV file
            output_file = self.output_dir / f"play_by_play_{year}.csv"
            pbp_data.to_csv(output_file, index=False)
            
            logger.info(f"Saved {len(pbp_data):,} plays to {output_file}")
            logger.info(f"Data shape: {pbp_data.shape}")
            
            # Log some basic statistics
            games = pbp_data['game_id'].nunique()
            teams = pd.concat([pbp_data['home_team'], pbp_data['away_team']]).nunique()
            logger.info(f"Season {year}: {games} games, {teams} teams")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading data for {year}: {e}")
            return False
    
    def download_multiple_seasons(self, years: List[int]) -> Tuple[int, int]:
        """Download play-by-play data for multiple seasons"""
        valid_years = self.validate_years(years)
        
        if not valid_years:
            logger.error("No valid years provided")
            return 0, 0
        
        logger.info(f"Starting download for {len(valid_years)} seasons: {valid_years}")
        
        successful = 0
        failed = 0
        
        for year in valid_years:
            if self.download_season(year):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Download complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def get_data_info(self, year: int) -> Optional[dict]:
        """Get information about downloaded data for a specific year"""
        data_file = self.output_dir / f"play_by_play_{year}.csv"
        
        if not data_file.exists():
            logger.warning(f"No data file found for {year}: {data_file}")
            return None
        
        try:
            # Read just the header and first few rows for info
            df = pd.read_csv(data_file, nrows=5)
            full_df = pd.read_csv(data_file)
            
            info = {
                'year': year,
                'file': str(data_file),
                'total_plays': len(full_df),
                'columns': len(df.columns),
                'sample_columns': list(df.columns[:10]),  # First 10 columns
                'games': full_df['game_id'].nunique() if 'game_id' in full_df.columns else 'Unknown',
                'file_size_mb': round(data_file.stat().st_size / (1024 * 1024), 2)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error reading data file for {year}: {e}")
            return None


def parse_years_argument(years_str: str) -> List[int]:
    """Parse comma-separated years string into list of integers"""
    try:
        return [int(year.strip()) for year in years_str.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid year format: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Download NFL play-by-play data using nfl_data_py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_play_by_play.py --year 1999
    python download_play_by_play.py --start_year 1999 --end_year 2001
    python download_play_by_play.py --years 1999,2000,2001
    python download_play_by_play.py --info 1999
        """
    )
    
    # Mutually exclusive group for different ways to specify years
    year_group = parser.add_mutually_exclusive_group(required=True)
    
    year_group.add_argument(
        '--year',
        type=int,
        help='Download data for a single year'
    )
    
    year_group.add_argument(
        '--start_year',
        type=int,
        help='Start year for range (use with --end_year)'
    )
    
    year_group.add_argument(
        '--years',
        type=parse_years_argument,
        help='Comma-separated list of years (e.g., 1999,2000,2001)'
    )
    
    year_group.add_argument(
        '--info',
        type=int,
        help='Show information about already downloaded data for a year'
    )
    
    parser.add_argument(
        '--end_year',
        type=int,
        help='End year for range (use with --start_year)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw/play_by_play',
        help='Output directory for data files (default: data/raw/play_by_play)'
    )
    
    args = parser.parse_args()
    
    # Validate range arguments
    if args.start_year and not args.end_year:
        parser.error("--end_year is required when using --start_year")
    if args.end_year and not args.start_year:
        parser.error("--start_year is required when using --end_year")
    
    # Initialize downloader
    downloader = PlayByPlayDownloader(args.output_dir)
    
    # Handle info request
    if args.info:
        info = downloader.get_data_info(args.info)
        if info:
            print(f"\nData Information for {args.info}:")
            print(f"File: {info['file']}")
            print(f"Total plays: {info['total_plays']:,}")
            print(f"Columns: {info['columns']}")
            print(f"Games: {info['games']}")
            print(f"File size: {info['file_size_mb']} MB")
            print(f"Sample columns: {', '.join(info['sample_columns'])}")
        return
    
    # Determine years to download
    if args.year:
        years_to_download = [args.year]
    elif args.start_year and args.end_year:
        years_to_download = list(range(args.start_year, args.end_year + 1))
    elif args.years:
        years_to_download = args.years
    else:
        parser.error("Must specify years to download")
    
    print("\n" + "="*60)
    print("NFL PLAY-BY-PLAY DATA DOWNLOADER")
    print("="*60)
    print(f"Years to download: {years_to_download}")
    print(f"Output directory: {downloader.output_dir}")
    print("="*60 + "\n")
    
    # Download the data
    successful, failed = downloader.download_multiple_seasons(years_to_download)
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total attempts: {successful + failed}")
    
    if successful > 0:
        print(f"\nData saved to: {downloader.output_dir}")
        print("Use --info <year> to view details about downloaded data")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()