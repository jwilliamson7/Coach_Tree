# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL coaching tree analysis project that models coaching relationships as a genetic tree. The system analyzes how coaching philosophies and play-calling patterns ("genes") propagate through the NFL coaching network. It combines coaching career data from Pro Football Reference with play-by-play data (via nfl_data_py) to understand how coaching strategies evolve and spread through mentor-protege relationships.

## Architecture

### Data Pipeline Flow
1. **Scraping** (`crawlers/PFR/`): Scrapes coach and team data from pro-football-reference.com
2. **Raw Data Storage** (`data/raw/`): Stores individual coach and team data files  
3. **Processing** (`scripts/`): Transforms raw data into feature-engineered datasets
4. **Processed Data** (`data/processed/`): Contains normalized league data and coaching performance metrics
5. **Analysis** (`analysis/`): Statistical analysis and modeling (to be implemented)

### Key Components

- **Coaching Tree Framework** (`scripts/build_coaching_tree.py`)
  - `Coach` class: Represents individual coaches with complete career timelines
  - `CoachingTree` class: Manages all coach relationships and builds the network
  - Creates parent-child relationships between coaches based on team assignments
  - Distinguishes between offensive/defensive/special teams coaching branches

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
  - Defines role classifications and exclusion rules (e.g., interim positions)

## Common Commands

### Build Coaching Tree
```bash
# Build the coaching tree with all relationships
python scripts/build_coaching_tree.py
```

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
- pandas (<2.0, for compatibility with nfl_data_py)
- numpy
- nfl_data_py (for play-by-play data from 1999 onwards)
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
- `coaching_tree/`: Coaching tree relationship data
  - `coaches.json`: All coaches with complete career timelines
  - `relationships.csv`: Parent-child relationships between coaches
  - `team_rosters.json`: Team coaching staffs by year

## Key Features Tracked

### Coaching Tree Relationships
- **Parent-child relationships** between coaches based on team assignments
- **Three relationship types**:
  - Position coach → Coordinator (931 relationships)
  - Position coach → Head Coach (2,353 relationships)  
  - Coordinator → Head Coach (1,552 relationships)
- **Role classifications**:
  - Head Coach (HC)
  - Offensive/Defensive/Special Teams Coordinator (OC/DC/STC)
  - Position coaches (classified by side of ball)

### Career Data
- Years of experience at different levels (NFL/College) and roles
- Previous performance metrics (win percentages, team rankings)
- Career progression patterns
- Hiring team context (recent performance, needs)
- Tenure classification (short/medium/long term success)

## Important Notes

### Coaching Tree Framework
- The `build_coaching_tree.py` script creates a complete network of coaching relationships
- Covers 536 coaches across 104 years of NFL history (1922-2025)
- Generates 4,836 documented relationships between coaches
- Parent assignments based on start-of-year positions (excludes interim roles)
- Handles team franchise mappings for accurate historical tracking

### Data Processing
- The `CoachingDataProcessor` class in `create_data.py` is the core processing engine that:
  - Classifies coaching roles and levels
  - Tracks career progression
  - Creates training instances for each head coaching hire
  - Calculates tenure outcomes

- Team franchise mappings in `data_constants.py` are critical for handling:
  - Team relocations (e.g., Oakland/Las Vegas Raiders)
  - Name changes (e.g., Houston Oilers/Tennessee Titans)
  - Historical teams no longer in existence

### Future Integration
- Play-by-play data (via nfl_data_py) will be used to generate coaching "genes"
- These genes will represent play-calling patterns and strategic tendencies
- Analysis will track how these genes propagate through the coaching tree
- Available play-by-play data covers 1999 onwards only