#!/usr/bin/env python3
"""
Model Features Selection for NFL Coaching Tree Analysis

This module defines quantitative and categorical fields suitable for predictive modeling
of coaching performance and play-calling patterns. Fields focus on game context,
situational factors, and strategic outcomes rather than individual player attributions.
"""

from typing import List, Dict, Set

# Game Context Fields
GAME_CONTEXT_FIELDS = [
    # Basic game info
    "season", "week", "season_type", "qtr", 
    
    # Time and score context
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "game_half", "quarter_end",
    
    # Score situation (pos/def team perspective only)
    "posteam_score", "defteam_score", "score_differential", 
    "posteam_score_post", "defteam_score_post", "score_differential_post",
    
    # Team identifiers (categorical)
    "posteam", "defteam", "posteam_type",
    
    # Game environment
    "location", "roof", "surface", "temp", "wind", "div_game",
]

# Situational Fields
SITUATIONAL_FIELDS = [
    # Down and distance
    "down", "ydstogo", "goal_to_go", "yardline_100",
    
    # Field position
    "side_of_field", "ydsnet",
    
    # Timeouts (pos/def team perspective only)
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "timeout",
    
    # Drive context
    "drive", "series", "series_success",
    "drive_play_count", "drive_time_of_possession", "drive_first_downs",
    "drive_inside20", "drive_ended_with_score", "drive_quarter_start", "drive_quarter_end",
    "drive_yards_penalized",
    
    # Special situations
    "two_point_attempt", "extra_point_attempt", "field_goal_attempt", 
    "kickoff_attempt", "punt_attempt",
]

# Pre-Play Formation Fields (only what's known before the play starts)
FORMATION_FIELDS = [
    # Formation and pre-snap (observable before play outcome)
    "shotgun", "no_huddle",
]

# Outcome Fields
OUTCOME_FIELDS = [
    # Basic outcomes
    "yards_gained", "first_down", "touchdown", "pass_touchdown", "rush_touchdown",
    "return_touchdown", "safety", "interception", "fumble", "fumble_lost",
    "incomplete_pass", "complete_pass", "sack", "tackled_for_loss",
    
    # Conversion outcomes
    "first_down_rush", "first_down_pass", "first_down_penalty",
    "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
    
    # Special teams outcomes
    "punt_blocked", "touchback", "punt_inside_twenty", "punt_in_endzone", 
    "punt_out_of_bounds", "punt_downed", "punt_fair_catch",
    "kickoff_inside_twenty", "kickoff_in_endzone", "kickoff_out_of_bounds",
    "kickoff_downed", "kickoff_fair_catch",
    
    # Other outcomes
    "penalty", "qb_hit", "own_kickoff_recovery", "own_kickoff_recovery_td",
    "fumble_forced", "fumble_not_forced", "fumble_out_of_bounds",
    "solo_tackle", "assist_tackle", "tackle_with_assist",
]

# Advanced Analytics Fields
ANALYTICS_FIELDS = [
    # Expected Points (pos/def team perspective only)
    "ep", "epa", "air_epa", "yac_epa", "comp_air_epa", "comp_yac_epa",
    "qb_epa", "xyac_epa", "xyac_mean_yardage", "xyac_median_yardage",
    
    # Win Probability (pos/def team perspective only)
    "wp", "def_wp", "wpa", "vegas_wpa", "vegas_wp",
    "air_wpa", "yac_wpa", "comp_air_wpa", "comp_yac_wpa",
    
    # Probability models
    "no_score_prob", "opp_fg_prob", "opp_safety_prob", "opp_td_prob",
    "fg_prob", "safety_prob", "td_prob", "extra_point_prob", "two_point_conversion_prob",
    "cp", "cpoe", "xpass", "pass_oe", "xyac_success", "xyac_fd",
    
    # Success metrics
    "success",
]

# Betting/Vegas Fields
BETTING_FIELDS = [
    "spread_line", "total_line", "home_opening_kickoff",
]

# Fields to EXCLUDE (player names, IDs, and descriptive text)
EXCLUDED_FIELDS = [
    # Player names and IDs
    "passer_player_name", "passer_player_id", "passer", "passer_id", "passer_jersey_number",
    "receiver_player_name", "receiver_player_id", "receiver", "receiver_id", "receiver_jersey_number", 
    "rusher_player_name", "rusher_player_id", "rusher", "rusher_id", "rusher_jersey_number",
    "td_player_name", "td_player_id", "interception_player_name", "interception_player_id",
    "punt_returner_player_name", "punt_returner_player_id", "kickoff_returner_player_name", "kickoff_returner_player_id",
    "punter_player_name", "punter_player_id", "kicker_player_name", "kicker_player_id",
    "blocked_player_name", "blocked_player_id", "safety_player_name", "safety_player_id",
    "penalty_player_name", "penalty_player_id",
    "name", "jersey_number", "id", "fantasy_player_name", "fantasy_player_id", "fantasy", "fantasy_id",
    
    # Lateral player tracking (many fields)
    "lateral_receiver_player_id", "lateral_receiver_player_name", "lateral_receiving_yards",
    "lateral_rusher_player_id", "lateral_rusher_player_name", "lateral_rushing_yards",
    "lateral_sack_player_id", "lateral_sack_player_name",
    "lateral_interception_player_id", "lateral_interception_player_name",
    "lateral_punt_returner_player_id", "lateral_punt_returner_player_name",
    "lateral_kickoff_returner_player_id", "lateral_kickoff_returner_player_name",
    
    # Tackle and defensive player tracking  
    "tackle_for_loss_1_player_id", "tackle_for_loss_1_player_name",
    "tackle_for_loss_2_player_id", "tackle_for_loss_2_player_name",
    "qb_hit_1_player_id", "qb_hit_1_player_name", "qb_hit_2_player_id", "qb_hit_2_player_name",
    "forced_fumble_player_1_player_id", "forced_fumble_player_1_player_name",
    "forced_fumble_player_2_player_id", "forced_fumble_player_2_player_name",
    "solo_tackle_1_player_id", "solo_tackle_1_player_name", "solo_tackle_2_player_id", "solo_tackle_2_player_name",
    "assist_tackle_1_player_id", "assist_tackle_1_player_name", "assist_tackle_2_player_id", "assist_tackle_2_player_name",
    "assist_tackle_3_player_id", "assist_tackle_3_player_name", "assist_tackle_4_player_id", "assist_tackle_4_player_name",
    "tackle_with_assist_1_player_id", "tackle_with_assist_1_player_name",
    "tackle_with_assist_2_player_id", "tackle_with_assist_2_player_name",
    "pass_defense_1_player_id", "pass_defense_1_player_name", "pass_defense_2_player_id", "pass_defense_2_player_name",
    "fumbled_1_player_id", "fumbled_1_player_name", "fumbled_2_player_id", "fumbled_2_player_name",
    "fumble_recovery_1_player_id", "fumble_recovery_1_player_name",
    "fumble_recovery_2_player_id", "fumble_recovery_2_player_name",
    "sack_player_id", "sack_player_name", "half_sack_1_player_id", "half_sack_1_player_name",
    "half_sack_2_player_id", "half_sack_2_player_name", "own_kickoff_recovery_player_id", "own_kickoff_recovery_player_name",
    
    # Team tracking for defensive plays (redundant with posteam/defteam)
    "forced_fumble_player_1_team", "forced_fumble_player_2_team", "solo_tackle_1_team", "solo_tackle_2_team",
    "assist_tackle_1_team", "assist_tackle_2_team", "assist_tackle_3_team", "assist_tackle_4_team",
    "tackle_with_assist_1_team", "tackle_with_assist_2_team", "fumbled_1_team", "fumbled_2_team",
    "fumble_recovery_1_team", "fumble_recovery_2_team", "return_team", "penalty_team", "timeout_team", "td_team",
    
    # Descriptive text and metadata
    "desc", "yrdln", "time", "game_date", "start_time", "time_of_day", "play_clock", "end_clock_time",
    "stadium", "weather", "stadium_id", "game_stadium",
    "home_coach", "away_coach",  # Coach names (could be useful but inconsistent format)
    "drive_real_start_time", "drive_game_clock_start", "drive_game_clock_end",
    "drive_start_transition", "drive_end_transition", "series_result",
    "drive_start_yard_line", "drive_end_yard_line", "end_yard_line", "drive_play_id_started", "drive_play_id_ended",
    
    # Technical/processing fields
    "play_id", "game_id", "old_game_id", "nfl_api_id", "order_sequence", "play_deleted", 
    "play_type_nfl", "st_play_type", "fixed_drive", "fixed_drive_result",
    "replay_or_challenge", "replay_or_challenge_result", "penalty_type",
    "passing_yards", "receiving_yards", "rushing_yards",  # Individual stats
    "fumble_recovery_1_yards", "fumble_recovery_2_yards", "return_yards", "penalty_yards",
    "lateral_reception", "lateral_rush", "lateral_return", "lateral_recovery",
    "defensive_two_point_attempt", "defensive_two_point_conv", "defensive_extra_point_attempt", "defensive_extra_point_conv",
    "aborted_play", "out_of_bounds",
    
    # Home/Away team stats (now redundant with pos/def team + home flag)
    "home_team", "away_team", "total_home_score", "total_away_score", 
    "home_timeouts_remaining", "away_timeouts_remaining",
    "total_home_epa", "total_away_epa", "total_home_rush_epa", "total_away_rush_epa", 
    "total_home_pass_epa", "total_away_pass_epa", "total_home_comp_air_epa", "total_away_comp_air_epa",
    "total_home_comp_yac_epa", "total_away_comp_yac_epa", "total_home_raw_air_epa", "total_away_raw_air_epa",
    "total_home_raw_yac_epa", "total_away_raw_yac_epa",
    "home_wp", "away_wp", "vegas_home_wpa", "home_wp_post", "away_wp_post", "vegas_home_wp",
    "total_home_rush_wpa", "total_away_rush_wpa", "total_home_pass_wpa", "total_away_pass_wpa",
    "total_home_comp_air_wpa", "total_away_comp_air_wpa", "total_home_comp_yac_wpa", "total_away_comp_yac_wpa",
    "total_home_raw_air_wpa", "total_away_raw_air_wpa", "total_home_raw_yac_wpa", "total_away_raw_yac_wpa",
    "result", "total", "away_score", "home_score",  # Game result/scores from home team perspective
]

# Note: The following fields are intentionally EXCLUDED to avoid redundancy:
# - All home_team/away_team specific stats (use posteam/defteam + posteam_type instead)
# - Individual player names and IDs (focus on game context, not individual performance)
# - Cumulative home/away EPA and WPA stats (use possession-based perspective)
# - Game result/total from home team perspective (use score differential instead)

def get_all_model_features() -> List[str]:
    """
    Get all fields suitable for model training.
    
    Returns:
        List of field names for model features
    """
    all_features = (
        GAME_CONTEXT_FIELDS + SITUATIONAL_FIELDS + FORMATION_FIELDS + 
        OUTCOME_FIELDS + ANALYTICS_FIELDS + BETTING_FIELDS
    )
    return sorted(list(set(all_features)))

def get_feature_groups() -> Dict[str, List[str]]:
    """
    Get features organized by category.
    
    Returns:
        Dictionary mapping category names to lists of field names
    """
    return {
        "game_context": GAME_CONTEXT_FIELDS,
        "situational": SITUATIONAL_FIELDS, 
        "formation": FORMATION_FIELDS,
        "outcomes": OUTCOME_FIELDS,
        "analytics": ANALYTICS_FIELDS,
        "betting": BETTING_FIELDS,
    }

def get_categorical_features() -> Set[str]:
    """
    Get features that should be treated as categorical variables.
    
    Returns:
        Set of field names that are categorical
    """
    return {
        # Team identifiers
        "posteam", "defteam", "posteam_type",
        
        # Game characteristics
        "season_type", "game_half", "location", "roof", "surface",
        
        # Play characteristics  
        "play_type", "pass_length", "pass_location", "run_location", "run_gap",
        "field_goal_result", "extra_point_result", "two_point_conv_result",
        
        # Field position
        "side_of_field",
        
        # Result categories
        "series_result", "drive_start_transition", "drive_end_transition",
    }

def get_predictor_features() -> List[str]:
    """
    Get features suitable as predictors (X variables) - excludes outcomes.
    Only includes pre-play context available to coaches when making decisions.
    
    Returns:
        List of field names suitable as predictor variables
    """
    predictors = (
        GAME_CONTEXT_FIELDS + SITUATIONAL_FIELDS + FORMATION_FIELDS + 
        BETTING_FIELDS
    )
    return sorted(list(set(predictors)))

def get_basic_predictor_features() -> List[str]:
    """
    Get basic predictor features excluding advanced analytics to avoid confounding.
    
    Use this for coaching pattern analysis where you want to identify decision-making
    from raw game context rather than derived metrics.
    
    Returns:
        List of basic field names for predictor variables
    """
    return get_predictor_features()  # Already excludes analytics

def get_enhanced_predictor_features() -> List[str]:
    """
    Get enhanced predictor features including selected advanced analytics.
    
    Use this for performance prediction where analytical context is valuable.
    Excludes post-play outcomes but includes pre-play probabilities and context.
    
    Returns:
        List of enhanced field names for predictor variables
    """
    # Safe analytics that don't leak outcome information
    safe_analytics = [
        # Pre-play probabilities (available before play outcome)
        "no_score_prob", "opp_fg_prob", "opp_safety_prob", "opp_td_prob",
        "fg_prob", "safety_prob", "td_prob", "extra_point_prob", "two_point_conversion_prob",
        "cp", "xpass", "xyac_success", "xyac_fd",
        
        # Pre-play context (not dependent on current play outcome) 
        "wp", "def_wp", "vegas_wp",
        "xyac_mean_yardage", "xyac_median_yardage",
    ]
    
    basic_features = get_predictor_features()
    return sorted(list(set(basic_features + safe_analytics)))

def get_fourth_down_predictor_features() -> List[str]:
    """
    Get features specifically for 4th down decision modeling.
    Only includes context available before the coaching decision is made.
    Excludes temporal fields (season, week) to focus on pure game context.
    
    Returns:
        List of field names for 4th down decision prediction
    """
    # Core context for 4th down decisions - NO temporal fields
    fourth_down_features = [
        # Game situation (excluding season/week)
        "qtr", 
        "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
        "game_half", 
        
        # Score situation  
        "posteam_score", "defteam_score", "score_differential",
        
        # Field position and down/distance
        "down", "ydstogo", "goal_to_go", "yardline_100", "side_of_field",
        
        # Timeouts and clock management
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        
        # Drive context (known at start of play)
        "drive_play_count", "drive_first_downs", "ydsnet",
        
        # Game environment
        "location", "div_game",
        
        # Vegas context
        "spread_line", "total_line",
    ]
    
    return sorted(fourth_down_features)

def get_run_pass_predictor_features() -> List[str]:
    """
    Get features specifically for run vs pass play type prediction.
    Only includes pre-play context available before the coaching decision is made.
    
    This excludes formation fields like 'shotgun' and 'no_huddle' since these
    are outcomes of the coaching decision we're trying to predict.
    Excludes temporal fields (season, week) to focus on pure game context.
    
    Returns:
        List of field names for run vs pass prediction
    """
    # Core context for run vs pass decisions - NO temporal fields
    run_pass_features = [
        # Game situation (excluding season/week)
        "qtr", 
        "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
        "game_half", 
        
        # Score situation  
        "posteam_score", "defteam_score", "score_differential",
        
        # Field position and down/distance  
        "down", "ydstogo", "goal_to_go", "yardline_100", "side_of_field",
        
        # Timeouts and clock management
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        
        # Drive context (known at start of play)
        "drive_play_count", "drive_first_downs", "ydsnet",
        
        # Game environment
        "location", "div_game",
        
        # Vegas context  
        "spread_line", "total_line",
    ]
    
    return sorted(run_pass_features)

def get_pass_target_predictor_features() -> List[str]:
    """
    Get features specifically for pass target prediction (behind vs ahead of first down marker).
    Only includes pre-play context available before the coaching decision is made.
    
    This excludes formation fields like 'shotgun' and 'no_huddle' since these
    are outcomes of the coaching decision we're trying to predict.
    Excludes temporal fields (season, week) to focus on pure game context.
    
    Returns:
        List of field names for pass target prediction
    """
    # Core context for pass target decisions - NO temporal fields
    pass_target_features = [
        # Game situation (excluding season/week)
        "qtr", 
        "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
        "game_half", 
        
        # Score situation  
        "posteam_score", "defteam_score", "score_differential",
        
        # Field position and down/distance  
        "down", "ydstogo", "goal_to_go", "yardline_100", "side_of_field",
        
        # Timeouts and clock management
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        
        # Drive context (known at start of play)
        "drive_play_count", "drive_first_downs", "ydsnet",
        
        # Game environment
        "location", "div_game",
        
        # Vegas context  
        "spread_line", "total_line",
    ]
    
    return sorted(pass_target_features)

def get_shotgun_predictor_features() -> List[str]:
    """
    Get features specifically for shotgun formation prediction.
    Only includes pre-play context available before the formation decision is made.
    
    This is similar to run/pass prediction but focuses on formation choice.
    Excludes temporal fields (season, week) to focus on pure game context.
    
    Returns:
        List of field names for shotgun formation prediction
    """
    # Core context for shotgun formation decisions - NO temporal fields
    shotgun_features = [
        # Game situation (excluding season/week)
        "qtr", 
        "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
        "game_half", 
        
        # Score situation  
        "posteam_score", "defteam_score", "score_differential",
        
        # Field position and down/distance  
        "down", "ydstogo", "goal_to_go", "yardline_100", "side_of_field",
        
        # Timeouts and clock management
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        
        # Drive context (known at start of play)
        "drive_play_count", "drive_first_downs", "ydsnet",
        
        # Game environment
        "location", "div_game",
        
        # Vegas context  
        "spread_line", "total_line",
    ]
    
    return sorted(shotgun_features)

def get_target_features() -> List[str]:
    """
    Get features suitable as targets (Y variables) - outcomes and analytics.
    
    Returns:
        List of field names suitable as target variables  
    """
    targets = OUTCOME_FIELDS + ANALYTICS_FIELDS
    return sorted(list(set(targets)))

def validate_features(available_columns: List[str]) -> Dict[str, List[str]]:
    """
    Validate which features are actually available in the dataset.
    
    Args:
        available_columns: List of column names in the actual dataset
        
    Returns:
        Dictionary with 'available' and 'missing' feature lists
    """
    all_features = get_all_model_features()
    available_set = set(available_columns)
    
    available_features = [f for f in all_features if f in available_set]
    missing_features = [f for f in all_features if f not in available_set]
    
    return {
        "available": available_features,
        "missing": missing_features,
        "total_features": len(all_features),
        "available_count": len(available_features), 
        "missing_count": len(missing_features)
    }

if __name__ == "__main__":
    # Print summary of features
    features = get_all_model_features()
    groups = get_feature_groups()
    categorical = get_categorical_features()
    predictors = get_predictor_features()
    targets = get_target_features()
    
    basic_predictors = get_basic_predictor_features()
    enhanced_predictors = get_enhanced_predictor_features()
    
    print("NFL PLAY-BY-PLAY MODEL FEATURES")
    print("=" * 50)
    print(f"Total features: {len(features)}")
    print(f"Categorical features: {len(categorical)}")
    print(f"Basic predictor features: {len(basic_predictors)}")
    print(f"Enhanced predictor features: {len(enhanced_predictors)}")
    print(f"Target features: {len(targets)}")
    print()
    
    for group_name, group_features in groups.items():
        print(f"{group_name.upper()}: {len(group_features)} features")
        for feature in group_features[:5]:  # Show first 5 
            marker = " (categorical)" if feature in categorical else ""
            print(f"  - {feature}{marker}")
        if len(group_features) > 5:
            print(f"  ... and {len(group_features) - 5} more")
        print()