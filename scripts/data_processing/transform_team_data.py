import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore
from typing import Dict, List


class TeamDataTransformer:
    """Transforms individual team data into league-wide yearly datasets"""
    
    # Columns that require special string parsing
    SPECIAL_COLUMNS = {
        'Start Average Drive': lambda x: float(x.split(' ')[-1]) if isinstance(x, str) else x,
        'Time Average Drive': lambda x: float(x.split(':')[0]) + float(x.split(':')[-1])/60 if isinstance(x, str) else x,
        '3D%': lambda x: float(x.rstrip('%')) / 100.0 if isinstance(x, str) else x,
        '4D%': lambda x: float(x.rstrip('%')) / 100.0 if isinstance(x, str) else x,
        'RZPct': lambda x: float(x.rstrip('%')) / 100.0 if isinstance(x, str) else x
    }
    
    def __init__(self, teams_dir: str = "../data/raw/Teams", output_dir: str = "../data/processed/League Data"):
        """Initialize transformer with input and output directories"""
        self.teams_dir = Path(teams_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _discover_team_directories(self) -> List[Path]:
        """Find all team directories"""
        return [d for d in self.teams_dir.iterdir() if d.is_dir()]
    
    def _load_team_data(self, team_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load team and opponent stats for a single team"""
        data = {}
        
        # Load team stats
        team_stats_file = team_dir / "yearly_team_stats.csv"
        if team_stats_file.exists():
            data['team'] = pd.read_csv(team_stats_file)
        
        # Load opponent stats  
        opponent_stats_file = team_dir / "yearly_opponent_stats.csv"
        if opponent_stats_file.exists():
            data['opponent'] = pd.read_csv(opponent_stats_file)
            
        return data
    
    def _process_team_data(self, df: pd.DataFrame, team_name: str) -> List[Dict]:
        """Process team data into list of year-based records"""
        records = []
        
        for _, row in df.iterrows():
            year = int(row['Year'])
            
            # Create record with team name and all stats (excluding original Year column)
            record = {'Team Abbreviation': team_name}
            
            # Add all other columns except the original Year and any index columns
            for col in df.columns:
                if col not in ['Year', 'Unnamed: 0']:  # Skip Year and any index columns
                    record[col] = row[col]
            
            records.append({'year': year, 'data': record})
            
        return records
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types for specific columns"""
        df_cleaned = df.copy()
        
        for col_name, converter in self.SPECIAL_COLUMNS.items():
            if col_name in df_cleaned.columns:
                # Apply conversion only to string/object dtype columns
                if df_cleaned[col_name].dtype == 'object':
                    df_cleaned[col_name] = df_cleaned[col_name].apply(converter)
        
        return df_cleaned
    
    def _save_datasets(self, df: pd.DataFrame, year_dir: Path, data_type: str) -> None:
        """Save both raw and normalized datasets"""
        # Save raw data
        raw_file = year_dir / f"league_{data_type}_data.csv"
        df.to_csv(raw_file, index=False)
        
        # Create normalized version
        df_normalized = df.copy()
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            df_normalized[numeric_cols] = df_normalized[numeric_cols].apply(zscore)
            
        # Save normalized data
        normalized_file = year_dir / f"league_{data_type}_data_normalized.csv"
        df_normalized.to_csv(normalized_file, index=False)
    
    def transform_all_teams(self) -> None:
        """Transform all team data into yearly league datasets"""
        team_dirs = self._discover_team_directories()
        
        # Organize data by year
        team_records_by_year = {}
        opponent_records_by_year = {}
        
        print(f"Processing {len(team_dirs)} teams...")
        
        for i, team_dir in enumerate(team_dirs, 1):
            team_name = team_dir.name
            print(f"Parsing team {i}: {team_name}")
            
            # Load team data
            team_data = self._load_team_data(team_dir)
            
            # Process team stats
            if 'team' in team_data:
                team_records = self._process_team_data(team_data['team'], team_name)
                for record in team_records:
                    year = record['year']
                    if year not in team_records_by_year:
                        team_records_by_year[year] = []
                    team_records_by_year[year].append(record['data'])
            
            # Process opponent stats
            if 'opponent' in team_data:
                opponent_records = self._process_team_data(team_data['opponent'], team_name)
                for record in opponent_records:
                    year = record['year']
                    if year not in opponent_records_by_year:
                        opponent_records_by_year[year] = []
                    opponent_records_by_year[year].append(record['data'])
        
        # Create and save yearly datasets
        print(f"\nCreating league datasets for {len(team_records_by_year)} years...")
        
        for year in sorted(team_records_by_year.keys()):
            year_dir = self.output_dir / str(year)
            year_dir.mkdir(exist_ok=True)
            
            # Process team data
            if year in team_records_by_year:
                team_df = pd.DataFrame(team_records_by_year[year])
                team_df = self._clean_data_types(team_df)
                self._save_datasets(team_df, year_dir, "team")
            
            # Process opponent data  
            if year in opponent_records_by_year:
                opponent_df = pd.DataFrame(opponent_records_by_year[year])
                opponent_df = self._clean_data_types(opponent_df)
                self._save_datasets(opponent_df, year_dir, "opponent")
            
            print(f"Created datasets for {year}")


def main():
    """Main execution function"""
    print("Starting Team Data Transformation...")
    
    # Initialize transformer
    transformer = TeamDataTransformer()
    
    # Transform all team data
    transformer.transform_all_teams()
    
    print("\nTransformation completed!")


if __name__ == "__main__":
    main()