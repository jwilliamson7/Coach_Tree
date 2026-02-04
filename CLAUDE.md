# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL coaching tree analysis project that models coaching relationships as a genetic tree. The system analyzes how coaching philosophies and play-calling patterns ("genes") propagate through the NFL coaching network. It combines coaching career data from Pro Football Reference with play-by-play data (via nfl_data_py) to understand how coaching strategies evolve and spread through mentor-protege relationships.

## Architecture

### Data Pipeline Flow
1. **Scraping** (`crawlers/PFR/`): Scrapes coach and team data from pro-football-reference.com
2. **Raw Data Storage** (`data/raw/`): Stores individual coach and team data files  
3. **Processing** (`scripts/data_processing/`): Transforms raw data into feature-engineered datasets
4. **Processed Data** (`data/processed/`): Contains normalized league data and coaching performance metrics
5. **Analysis** (`scripts/analysis/`): Gene calculations and behavioral analysis
6. **Visualization** (`scripts/visualization/`): Interactive charts and network graphs
7. **Generated Outputs** (`outputs/`): HTML visualizations, reports, and analysis results

### Key Components

- **Coaching Tree Framework** (`scripts/data_processing/build_coaching_tree.py`)
  - `Coach` class: Represents individual coaches with complete career timelines
  - `CoachingTree` class: Manages all coach relationships and builds the network
  - Creates parent-child relationships between coaches based on team assignments
  - Distinguishes between offensive/defensive/special teams coaching branches

- **Data Scrapers** (`crawlers/PFR/`)
  - `coach_scraping.py`: Scrapes individual coach career histories, results, and rankings
  - `team_data_scraping.py`: Scrapes team performance statistics by year

- **Data Processing Scripts** (`scripts/data_processing/`)
  - `create_data.py`: Main processing pipeline that creates master coaching dataset with engineered features
  - `transform_team_data.py`: Transforms individual team data into league-wide yearly datasets
  - `create_yearly_coach_performance_data.py`: Generates yearly coaching performance metrics
  - `extract_head_coaches.py`: Extracts and maps head coaching records
  - `build_coaching_tree.py`: Creates coaching tree relationships and network structure

- **Predictive Models** (`scripts/models/`)
  - `fourth_down_decision_model.py`: XGBoost model predicting 4th down go/no-go decisions
  - `run_pass_prediction_model.py`: XGBoost model predicting run vs pass play calls
  - `pass_target_prediction_model.py`: XGBoost model predicting pass targets behind vs ahead of first down marker
  - `two_point_conversion_model.py`: XGBoost model predicting two-point conversion vs extra point decisions

- **Coaching Gene Analysis** (`scripts/analysis/`)
  - `calculate_aggression_gene.py`: Calculates "aggression gene" for NFL coaches based on play-calling tendencies
  - `calculate_shotgun_gene.py`: Analyzes shotgun formation usage patterns
  - Compares actual decisions to model predictions to measure deviation from expected behavior
  - Four aggression components:
    - **4th Down Aggression**: Going for it on 4th down more/less than predicted
    - **Pass-Heavy Aggression**: Passing more/less than predicted in neutral situations
    - **Deep Pass Aggression**: Targeting beyond the sticks more/less than predicted
    - **Two-Point Aggression**: Attempting two-point conversions more/less than predicted
  - Generates composite aggression score combining all four dimensions
  - Handles team abbreviation mapping between different data sources
  - Processes ~900K plays per full run (2006-2024)

- **Visualization Tools** (`scripts/visualization/`)
  - `visualize_coaching_tree_aggression.py`: Interactive coaching tree with aggression gene overlay
  - `visualize_aggression_propagation.py`: Analysis of how aggression genes propagate through lineages
  - `visualize_coaching_tree_nx.py`: NetworkX-based coaching tree visualization with multiple layout options

- **Utilities** (`scripts/utils/`)
  - `svd_imputation.py`: Handles missing data imputation using SVD
  - `test_no_play_classification.py`: Testing utilities for play classification

- **Constants** (`crawlers/utils/data_constants.py`)
  - Contains all team abbreviation mappings, franchise histories, and configuration constants
  - Critical for handling team relocations and name changes throughout NFL history
  - Defines role classifications and exclusion rules (e.g., interim positions)

## Common Commands

### Build Coaching Tree
```bash
# Build the coaching tree with all relationships
python scripts/data_processing/build_coaching_tree.py
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
python scripts/data_processing/create_data.py

# Transform team data to league format
python scripts/data_processing/transform_team_data.py

# Generate yearly coach performance data
python scripts/data_processing/create_yearly_coach_performance_data.py --input_dir data/raw/Coaches --output_dir data/processed/Coaching

# Extract head coach mappings
python scripts/data_processing/extract_head_coaches.py --input_dir data/raw/Coaches --output_dir data/processed/Coaching
```

### Predictive Modeling
```bash
# Train 4th down decision prediction model
python "scripts/models/fourth_down_decision_model.py"

# Train run vs pass play type prediction model
python "scripts/models/run_pass_prediction_model.py"

# Train pass target prediction model (behind vs ahead of first down marker)
python "scripts/models/pass_target_prediction_model.py"

# Train two-point conversion decision model
python "scripts/models/two_point_conversion_model.py"
```

### Coaching Gene Analysis
```bash
# Calculate aggression gene for all coaches (2006-2024 by default)
python scripts/analysis/calculate_aggression_gene.py

# Specify custom year range
python scripts/analysis/calculate_aggression_gene.py --start_year 2010 --end_year 2023

# Custom output directory
python scripts/analysis/calculate_aggression_gene.py --output_dir data/processed/custom_genes
```

### Visualization
```bash
# Create coaching tree visualization with aggression gene overlay
python scripts/visualization/visualize_coaching_tree_aggression.py

# Create aggression gene propagation analysis
python scripts/visualization/visualize_aggression_propagation.py

# Create NetworkX-based coaching tree visualization
python scripts/visualization/visualize_coaching_tree_nx.py --layout kamada --color_by pagerank
```

### Missing Data Handling
```bash
# Impute missing values using SVD
python scripts/utils/svd_imputation.py --input data/processed/coaching_data.csv --output data/processed/coaching_data_imputed.csv
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
- xgboost (for predictive modeling)
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
  - `team_year_head_coaches.csv`: Mapping of teams and years to head coaches
- `coaching_tree/`: Coaching tree relationship data
  - `coaches.json`: All coaches with complete career timelines
  - `relationships.csv`: Parent-child relationships between coaches
  - `team_rosters.json`: Team coaching staffs by year
- `coaching_genes/`: Coaching behavioral analysis outputs
  - `aggression_gene_YYYYMMDD.csv`: Full aggression metrics for all coaches
  - `aggression_gene_summary_YYYYMMDD.json`: Summary statistics and rankings

### Generated Outputs (`outputs/`)
- `visualizations/`: Interactive HTML visualizations
  - `coaching_tree_*.html`: Various coaching tree network visualizations
  - `aggression_propagation.html`: Gene propagation analysis charts
- `reports/`: Analysis reports and summaries (future)
- `analysis/`: Detailed analysis outputs (future)

### Models (`models/`)
- `fourth_down/`: 4th down decision prediction model files
  - `fourth_down_decision_model.json`: Trained XGBoost model
  - `fourth_down_decision_model_metadata.json`: Model metadata and parameters
  - `fourth_down_decision_model_encoders.pkl`: Label encoders for categorical features
- `run_pass/`: Run vs pass prediction model files
  - `run_pass_prediction_model.json`: Trained XGBoost model
  - `run_pass_prediction_model_metadata.json`: Model metadata and parameters
  - `run_pass_prediction_model_encoders.pkl`: Label encoders for categorical features
- `pass_target/`: Pass target prediction model files
  - `pass_target_prediction_model.json`: Trained XGBoost model
  - `pass_target_prediction_model_metadata.json`: Model metadata and parameters
  - `pass_target_prediction_model_encoders.pkl`: Label encoders for categorical features
- `two_point/`: Two-point conversion decision model files
  - `two_point_conversion_model.json`: Trained XGBoost model
  - `two_point_conversion_model_metadata.json`: Model metadata and parameters
  - `two_point_conversion_model_encoders.pkl`: Label encoders for categorical features

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

### Predictive Models
- **4th Down Decisions**: Predicts go/no-go decisions on 4th down using game context (score, field position, time)
- **Run vs Pass**: Predicts play-calling tendencies based on situational factors
- **Pass Target Strategy**: Predicts whether passes target behind or ahead of the first down marker
- **Two-Point Conversions**: Predicts whether teams attempt two-point conversions vs extra points after touchdowns

## Important Notes

### Team Abbreviation Mapping
- **Critical**: Play-by-play data and coach mapping data use different team abbreviations
- The centralized `standardize_team_abbreviation()` function in `data_constants.py` handles all mappings
- Key mappings include: GB→GNB, KC→KAN, LA→LAR, SF→SFO, TB→TAM, etc.
- Without proper mapping, only ~55% of plays can be attributed to coaches
- With mapping, coverage increases to ~93% (remaining gaps are special plays without possession)
- **NOTE: The team abbreviation mapping is optimized for 2006+ data** (when play-by-play air_yards became available). Pre-2006 historical team mappings (early NFL era, defunct franchises) may not be fully accurate. If extending analysis to earlier years, review and expand `FULL_TEAM_NAME_TO_PFR_ABBREV` in `data_constants.py`.

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

### Predictive Modeling Framework
- XGBoost models use only pre-play context features to avoid data leakage
- **Environmental factors included**: Weather (temperature, wind), field conditions (roof, surface), location (home/neutral)
- **Temporal fields excluded**: Season and week removed to prevent confounding in gene analysis
- SVD-based imputation handles missing values in large datasets
- RandomizedSearchCV with stratified cross-validation for hyperparameter tuning
- Models saved in XGBoost native JSON format for optimal performance
- Label encoders preserve categorical feature mappings
- Current model performance:
  - 4th Down Decisions: AUC 0.968 (26 features)
  - Run vs Pass: AUC 0.786 (26 features)
  - Pass Target: AUC 0.732 (26 features)
  - Two-Point Conversion: AUC 0.927 (21 features)

### Future Integration
- Play-by-play data (via nfl_data_py) will be used to generate coaching "genes"
- These genes will represent play-calling patterns and strategic tendencies
- Analysis will track how these genes propagate through the coaching tree
- Current models establish baseline coaching decision patterns
- Available play-by-play data covers 1999 onwards only (air_yards from 2006+)