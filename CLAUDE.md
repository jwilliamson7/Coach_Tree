# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL coaching data analysis project that collects, processes, and analyzes coaching career data from Pro Football Reference. The system tracks coaching progression through various roles (Head Coach, Coordinators, Position coaches) across NFL and college careers, creating a comprehensive dataset for analyzing coaching career trajectories and performance.

## Architecture

### Data Pipeline Flow
1. **Scraping** (`crawlers/PFR/`): Scrapes coach and team data from pro-football-reference.com
2. **Raw Data Storage** (`data/raw/`): Stores individual coach and team data files  
3. **Processing** (`scripts/`): Transforms raw data into feature-engineered datasets
4. **Processed Data** (`data/processed/`): Contains normalized league data and coaching performance metrics
5. **Analysis** (`analysis/`): Statistical analysis and modeling (to be implemented)

### Key Components

- **Data Scrapers** (`crawlers/PFR/`)
  - `coach_scraping.py`: Scrapes individual coach career histories, results, and rankings
  - `team_data_scraping.py`: Scrapes team performance statistics by year

- **Processing Scripts** (`scripts/`)
  - `create_data.py`: Main processing pipeline that creates master coaching dataset with engineered features
  - `transform_team_data.py`: Transforms individual team data into league-wide yearly datasets
  - `create_yearly_coach_performance_data.py`: Generates yearly coaching performance metrics
  - `extract_head_coaches.py`: Extracts and maps head coaching records
  - `svd_imputation.py`: Handles missing data imputation using SVD

- **Constants** (`crawlers/utils/data_constants.py`)
  - Contains all team abbreviation mappings, franchise histories, and configuration constants
  - Critical for handling team relocations and name changes throughout NFL history

## Common Commands

### Data Collection
```bash
# Scrape coach data
python crawlers/PFR/coach_scraping.py

# Scrape team data  
python crawlers/PFR/team_data_scraping.py
```

### Data Processing
```bash
# Create master coaching dataset
python scripts/create_data.py

# Transform team data to league format
python scripts/transform_team_data.py

# Generate yearly coach performance data
python scripts/create_yearly_coach_performance_data.py --input_dir data/raw/Coaches --output_dir data/processed/Coaching

# Extract head coach mappings
python scripts/extract_head_coaches.py --input_dir data/raw/Coaches --output_dir data/processed/Coaching
```

### Missing Data Handling
```bash
# Impute missing values using SVD
python scripts/svd_imputation.py --input data/processed/coaching_data.csv --output data/processed/coaching_data_imputed.csv
```

## Dependencies

Required Python packages:
- pandas
- numpy
- requests
- beautifulsoup4
- lxml
- scipy
- scikit-learn
- pathlib

## Data Structure

### Raw Data (`data/raw/`)
- `Coaches/{coach_name}/`: Individual coach directories containing:
  - `all_coaching_history.csv`: Career history with teams and roles
  - `all_coaching_results.csv`: Win-loss records by year
  - `all_coaching_ranks.csv`: Team rankings during tenure
- `Teams/{team_abbrev}/`: Team statistics and records

### Processed Data (`data/processed/`)
- `League Data/{year}/`: Yearly league-wide statistics
  - `league_team_data.csv`: Team offensive statistics
  - `league_opponent_data.csv`: Defensive statistics (opponent performance)
  - `*_normalized.csv`: Z-score normalized versions
- `Coaching/`: Processed coaching datasets with features and outcomes

## Key Features Tracked

The system tracks comprehensive coaching features including:
- Years of experience at different levels (NFL/College) and roles
- Previous performance metrics (win percentages, team rankings)
- Career progression patterns
- Hiring team context (recent performance, needs)
- Tenure classification (short/medium/long term success)

## Important Notes

- The `CoachingDataProcessor` class in `create_data.py` is the core processing engine that:
  - Classifies coaching roles and levels
  - Tracks career progression
  - Creates training instances for each head coaching hire
  - Calculates tenure outcomes

- Team franchise mappings in `data_constants.py` are critical for handling:
  - Team relocations (e.g., Oakland/Las Vegas Raiders)
  - Name changes (e.g., Houston Oilers/Tennessee Titans)
  - Historical teams no longer in existence

- Data spans from 1920 to present, requiring careful handling of rule changes and statistical availability across eras

- Excluded instances and special cases (fired coaches, interim roles) are handled via configuration in `data_constants.py`