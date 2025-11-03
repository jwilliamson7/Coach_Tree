"""
Coach Background Analysis from Coaching History
Uses actual coaching history data from pro-football-reference to classify coaches
by their coordinator background and analyze WAR performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy import stats

def load_all_coaching_histories():
    """Load all coaching history files to analyze backgrounds."""
    
    all_histories = []
    coach_dirs = glob.glob('data/raw/Coaches/*')
    
    print(f"Found {len(coach_dirs)} coach directories")
    
    for coach_dir in coach_dirs:
        coach_name = os.path.basename(coach_dir)
        history_file = os.path.join(coach_dir, 'all_coaching_history.csv')
        
        if os.path.exists(history_file):
            try:
                history_df = pd.read_csv(history_file)
                history_df['Coach_Directory'] = coach_name
                all_histories.append(history_df)
            except Exception as e:
                print(f"Error reading {history_file}: {e}")
                continue
    
    if all_histories:
        combined_histories = pd.concat(all_histories, ignore_index=True)
        print(f"Loaded coaching histories for {len(all_histories)} coaches")
        return combined_histories
    else:
        print("No coaching history files found!")
        return pd.DataFrame()

def classify_coach_background(coach_histories):
    """Classify each coach's background based on their coordinator experience."""
    
    # Get unique coaches
    coaches = coach_histories['Coach_Directory'].unique()
    coach_backgrounds = []
    
    print(f"\nAnalyzing {len(coaches)} coaches...")
    
    for coach_dir in coaches:
        coach_data = coach_histories[coach_histories['Coach_Directory'] == coach_dir]
        
        # Use the directory name as the coach identifier for now
        # We'll need to map this to the actual names later
        coach_name = coach_dir.replace('_', ' ')
        
        # Look for coordinator and position coach roles before head coaching
        # Filter out head coach roles to focus on coordinator/position coach background
        non_hc_roles = coach_data[~coach_data['Role'].str.contains('Head Coach', case=False, na=False)]
        
        # Look for offensive coordinator and position coach experience
        offensive_patterns = [
            'Offensive Coordinator', 'Off\. Coordinator',
            'Quarterbacks', 'QB Coach', 'Quarterback',
            'Running Backs', 'RB Coach', 'Running Back',
            'Wide Receivers', 'WR Coach', 'Wide Receiver', 'Receivers',
            'Tight Ends', 'TE Coach', 'Tight End',
            'Offensive Line', 'OL Coach', 'O-Line',
            'Offensive Quality Control', 'Offensive Assistant',
            'Offensive Backs', 'Passing Game Coordinator',
            'Run Game Coordinator', 'Offensive Intern',
            'Line Coach', 'Backfield', 'Offensive Backfield',
            'Ends Coach', 'Centers', 'Guards',
            'Tackles Coach', 'Receivers Coach', 'Backs Coach', 'Tackles',
            'Offensive Coach', 'Play-Caller', 'Ends/Centers'
        ]
        offensive_pattern = '|'.join(offensive_patterns)
        offensive_experience = non_hc_roles[non_hc_roles['Role'].str.contains(
            offensive_pattern, case=False, na=False)]
        
        # Look for defensive coordinator and position coach experience
        defensive_patterns = [
            'Defensive Coordinator', 'Def\. Coordinator',
            'Linebackers', 'LB Coach', 'Linebacker',
            'Defensive Backs', 'DB Coach', 'Secondary',
            'Cornerbacks', 'CB Coach', 'Cornerback',
            'Safeties', 'Safety',
            'Defensive Line', 'DL Coach', 'D-Line',
            'Defensive Ends', 'DE Coach',
            'Defensive Tackles', 'DT Coach',
            'Defensive Quality Control', 'Defensive Assistant',
            'Pass Rush Specialist', 'Defensive Intern',
            'Defensive Backfield', 'Defensive Coach'
        ]
        defensive_pattern = '|'.join(defensive_patterns)
        defensive_experience = non_hc_roles[non_hc_roles['Role'].str.contains(
            defensive_pattern, case=False, na=False)]
        
        # Count years of experience
        offensive_years = len(offensive_experience)
        defensive_years = len(defensive_experience)
        
        # Classify background
        if offensive_years > 0 and defensive_years > 0:
            # If both, classify by which is more prominent
            if offensive_years > defensive_years:
                background = 'Offensive'
            elif defensive_years > offensive_years:
                background = 'Defensive'
            else:
                background = 'Both'
        elif offensive_years > 0:
            background = 'Offensive'
        elif defensive_years > 0:
            background = 'Defensive'
        else:
            background = 'Other'
        
        # Get some sample roles for verification
        all_roles = coach_data['Role'].dropna().unique()
        sample_roles = ', '.join(all_roles[:3]) if len(all_roles) > 0 else 'No roles found'
        
        coach_backgrounds.append({
            'Coach_Directory': coach_dir,
            'Coach_Name': coach_name,
            'Background': background,
            'Offensive_Years': offensive_years,
            'Defensive_Years': defensive_years,
            'Total_Coaching_Years': len(coach_data),
            'Sample_Roles': sample_roles
        })
    
    return pd.DataFrame(coach_backgrounds)

def match_coaches_with_war_data(coach_backgrounds, impact_df):
    """Match coaching backgrounds with WAR performance data."""
    
    # Create various name matching strategies
    matched_data = []
    unmatched_coaches = []
    
    for _, bg_row in coach_backgrounds.iterrows():
        coach_name = bg_row['Coach_Name']
        
        # Try exact match first
        war_matches = impact_df[impact_df['Primary_Coach'] == coach_name]
        
        # If no exact match, try partial matching
        if len(war_matches) == 0:
            # Try matching by last name
            last_name = coach_name.split()[-1] if ' ' in coach_name else coach_name
            war_matches = impact_df[impact_df['Primary_Coach'].str.contains(last_name, case=False, na=False)]
        
        if len(war_matches) > 0:
            # Add background info to each season
            war_matches = war_matches.copy()
            war_matches['Background'] = bg_row['Background']
            war_matches['Offensive_Years'] = bg_row['Offensive_Years']
            war_matches['Defensive_Years'] = bg_row['Defensive_Years']
            matched_data.append(war_matches)
        else:
            unmatched_coaches.append(coach_name)
    
    if matched_data:
        combined_data = pd.concat(matched_data, ignore_index=True)
        print(f"\nMatched {len(combined_data)} coach-seasons")
        print(f"Unmatched coaches: {len(unmatched_coaches)}")
        if len(unmatched_coaches) < 20:  # Don't print too many
            print(f"Sample unmatched: {unmatched_coaches[:10]}")
        return combined_data
    else:
        print("No matches found between coaching histories and WAR data!")
        return pd.DataFrame()

def calculate_cumulative_war_by_background(matched_data):
    """Calculate average cumulative WAR trajectories by coach background."""
    
    # Calculate cumulative WAR for each coach
    coach_trajectories = []
    
    for coach in matched_data['Primary_Coach'].unique():
        coach_data = matched_data[matched_data['Primary_Coach'] == coach].sort_values('Year')
        if len(coach_data) == 0:
            continue
            
        # Calculate cumulative WAR in games
        coach_data = coach_data.copy()
        coach_data['Cumulative_WAR_Games'] = (coach_data['Coaching_WAR'].cumsum() * 16)
        coach_data['Season_Number'] = range(1, len(coach_data) + 1)
        
        background = coach_data['Background'].iloc[0]
        
        for _, row in coach_data.iterrows():
            coach_trajectories.append({
                'Coach': coach,
                'Background': background,
                'Season_Number': row['Season_Number'],
                'Cumulative_WAR_Games': row['Cumulative_WAR_Games'],
                'Single_Season_WAR': row['Coaching_WAR'] * 16
            })
    
    return pd.DataFrame(coach_trajectories)

def create_background_war_analysis():
    """Create comprehensive analysis of coach backgrounds and WAR performance."""
    
    # Load coaching histories
    print("Loading coaching history data...")
    coach_histories = load_all_coaching_histories()
    
    if coach_histories.empty:
        print("No coaching history data found!")
        return None, None, None
    
    # Classify coach backgrounds
    print("Classifying coach backgrounds from actual coaching history...")
    coach_backgrounds = classify_coach_background(coach_histories)
    
    # Print background distribution
    background_counts = coach_backgrounds['Background'].value_counts()
    print(f"\nCoach Background Distribution (from actual history):")
    print("=" * 50)
    for background, count in background_counts.items():
        pct = (count / len(coach_backgrounds)) * 100
        print(f"{background:<12}: {count:3d} coaches ({pct:5.1f}%)")
    
    # Load WAR data
    print("\nLoading coaching impact analysis data...")
    impact_df = pd.read_csv('data/final/coaching_impact_analysis.csv')
    
    # Match backgrounds with WAR data
    print("Matching coach backgrounds with WAR performance data...")
    matched_data = match_coaches_with_war_data(coach_backgrounds, impact_df)
    
    # Filter out "Both" category for cleaner analysis
    if not matched_data.empty:
        matched_data = matched_data[matched_data['Background'] != 'Both']
        print(f"Filtered out 'Both' category, remaining: {len(matched_data)} coach-seasons")
    
    if matched_data.empty:
        print("No matches found!")
        return coach_backgrounds, None, None
    
    # Calculate trajectories
    print("Calculating cumulative WAR trajectories...")
    trajectories_df = calculate_cumulative_war_by_background(matched_data)
    
    # Limit to first 15 seasons for trajectory analysis
    trajectories_limited = trajectories_df[trajectories_df['Season_Number'] <= 15]
    
    # Calculate average trajectories by background and season (first 15 seasons only)
    avg_trajectories = trajectories_limited.groupby(['Background', 'Season_Number']).agg({
        'Cumulative_WAR_Games': ['mean', 'std', 'count'],
        'Single_Season_WAR': 'mean'
    }).round(2)
    
    # Flatten column names
    avg_trajectories.columns = ['Avg_Cumulative_WAR', 'Std_Cumulative_WAR', 'Coach_Count', 'Avg_Single_Season_WAR']
    avg_trajectories = avg_trajectories.reset_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors for backgrounds
    colors = {
        'Offensive': '#FF6B35',  # Orange
        'Defensive': '#004E89',  # Dark Blue  
        'Other': '#808080'       # Gray
    }
    
    # Plot Average Cumulative WAR Trajectories
    for background in avg_trajectories['Background'].unique():
        if pd.isna(background):
            continue
            
        bg_data = avg_trajectories[avg_trajectories['Background'] == background]
        matched_count = len(matched_data[matched_data['Background'] == background]['Primary_Coach'].unique())
        
        ax.plot(bg_data['Season_Number'], bg_data['Avg_Cumulative_WAR'], 
                color=colors.get(background, '#000000'), linewidth=3, 
                label=f"{background} ({matched_count} coaches)",
                marker='o', markersize=6, alpha=0.8)
        
        # Add confidence bands (±1 std)
        if len(bg_data) > 1:
            ax.fill_between(bg_data['Season_Number'], 
                           bg_data['Avg_Cumulative_WAR'] - bg_data['Std_Cumulative_WAR'],
                           bg_data['Avg_Cumulative_WAR'] + bg_data['Std_Cumulative_WAR'],
                           color=colors.get(background, '#000000'), alpha=0.2)
    
    # Customize plot
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Season Number in Career', fontsize=16, family='Cambria')
    ax.set_ylabel('Average Cumulative WAR (Wins Above Replacement)', fontsize=16, family='Cambria')
    ax.set_title('Average Cumulative WAR Trajectories by Actual Coach Background (First 15 Seasons)', fontsize=18, fontweight='bold', family='Cambria')
    ax.grid(True, alpha=0.3)
    legend = ax.legend(fontsize=12, loc='upper left')
    for text in legend.get_texts():
        text.set_family('Cambria')
    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family('Cambria')
    
    plt.tight_layout()
    
    # Set x-axis to show seasons 1-15
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(range(1, 16))
    
    # Save the plot
    output_file = 'analysis/outputs/png/coach_background_from_history_15seasons.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nPlot saved as: {output_file}")
    
    # Print detailed statistics
    print(f"\nDetailed Background Analysis (based on actual coaching history):")
    print("=" * 70)
    
    for background in ['Offensive', 'Defensive', 'Other']:
        bg_matched = matched_data[matched_data['Background'] == background]
        bg_traj = trajectories_df[trajectories_df['Background'] == background]
        
        if len(bg_traj) > 0:
            unique_coaches = len(bg_matched['Primary_Coach'].unique())
            avg_career_war = bg_traj.groupby('Coach')['Cumulative_WAR_Games'].max().mean()
            avg_single_war = bg_traj['Single_Season_WAR'].mean()
            
            print(f"\n{background} Coaches ({unique_coaches} matched with WAR data):")
            print(f"  Average career WAR: {avg_career_war:+.1f} games")
            print(f"  Average single season WAR: {avg_single_war:+.2f} games")
            
            # Top 3 coaches by career WAR
            career_wars = bg_traj.groupby('Coach')['Cumulative_WAR_Games'].max().sort_values(ascending=False)
            print(f"  Top 3 by career WAR:")
            for coach, war in career_wars.head(3).items():
                print(f"    {coach}: {war:+.1f} games")
                
            # Sample coaching experience for verification
            bg_coaches = coach_backgrounds[coach_backgrounds['Background'] == background]
            if len(bg_coaches) > 0:
                sample_coach = bg_coaches.iloc[0]
                print(f"  Sample: {sample_coach['Coach_Name']} had {sample_coach['Offensive_Years']} offensive years, {sample_coach['Defensive_Years']} defensive years")
    
    # Statistical comparison between Offensive and Defensive coaches
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISON: OFFENSIVE vs DEFENSIVE COACHES")
    print(f"{'='*70}")
    
    # First, calculate season numbers for each coach
    matched_data_with_seasons = matched_data.copy()
    matched_data_with_seasons = matched_data_with_seasons.sort_values(['Primary_Coach', 'Year'])
    matched_data_with_seasons['Season_Number'] = matched_data_with_seasons.groupby('Primary_Coach').cumcount() + 1
    
    # Filter to only first 15 seasons
    first_15_seasons = matched_data_with_seasons[matched_data_with_seasons['Season_Number'] <= 15]
    
    print(f"\nFiltering to first 15 seasons of each coach's career...")
    print(f"Original dataset: {len(matched_data)} coach-seasons")
    print(f"First 15 seasons only: {len(first_15_seasons)} coach-seasons")
    
    offensive_war = first_15_seasons[first_15_seasons['Background'] == 'Offensive']['Coaching_WAR']
    defensive_war = first_15_seasons[first_15_seasons['Background'] == 'Defensive']['Coaching_WAR']
    
    # Count unique coaches in each category
    offensive_coaches = first_15_seasons[first_15_seasons['Background'] == 'Offensive']['Primary_Coach'].nunique()
    defensive_coaches = first_15_seasons[first_15_seasons['Background'] == 'Defensive']['Primary_Coach'].nunique()
    
    if len(offensive_war) > 0 and len(defensive_war) > 0:
        # Descriptive statistics
        print(f"\nDescriptive Statistics (First 15 Seasons Only):")
        print(f"Offensive Coaches:")
        print(f"  Unique coaches: {offensive_coaches}")
        print(f"  Sample size: {len(offensive_war)} coach-seasons")
        print(f"  Mean WAR: {offensive_war.mean():+.4f}")
        print(f"  Std Dev: {offensive_war.std():.4f}")
        print(f"  Median WAR: {offensive_war.median():+.4f}")
        
        print(f"\nDefensive Coaches:")
        print(f"  Unique coaches: {defensive_coaches}")
        print(f"  Sample size: {len(defensive_war)} coach-seasons")
        print(f"  Mean WAR: {defensive_war.mean():+.4f}")
        print(f"  Std Dev: {defensive_war.std():.4f}")
        print(f"  Median WAR: {defensive_war.median():+.4f}")
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(offensive_war, defensive_war, equal_var=False)
        
        print(f"\nTwo-Sample T-Test (Welch's t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(offensive_war) - 1) * offensive_war.var() + 
                             (len(defensive_war) - 1) * defensive_war.var()) / 
                            (len(offensive_war) + len(defensive_war) - 2))
        cohens_d = (offensive_war.mean() - defensive_war.mean()) / pooled_std
        
        print(f"  Cohen's d (effect size): {cohens_d:.4f}")
        
        # Interpretation
        if p_value < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        elif p_value < 0.10:
            significance = "marginally significant (p < 0.10)"
        else:
            significance = "not significant (p >= 0.10)"
        
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        mean_diff = offensive_war.mean() - defensive_war.mean()
        direction = "higher" if mean_diff > 0 else "lower"
        
        print(f"\nInterpretation:")
        print(f"  The difference is {significance}")
        print(f"  Effect size is {effect_size} (|d| = {abs(cohens_d):.4f})")
        print(f"  Offensive coaches have {direction} average WAR by {abs(mean_diff):.4f}")
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p_value = stats.mannwhitneyu(offensive_war, defensive_war, alternative='two-sided')
        
        print(f"\nMann-Whitney U Test (non-parametric):")
        print(f"  U-statistic: {u_stat:.0f}")
        print(f"  p-value: {u_p_value:.6f}")
        
        if u_p_value < 0.05:
            u_significance = "significant"
        else:
            u_significance = "not significant"
        print(f"  Non-parametric test: {u_significance}")
        
    else:
        print("Insufficient data for statistical comparison")
    
    # Save detailed data
    coach_backgrounds.to_csv('analysis/outputs/csv/coach_backgrounds_from_history.csv', index=False)
    avg_trajectories.to_csv('analysis/outputs/csv/coach_background_trajectories_from_history_15seasons.csv', index=False)
    matched_data.to_csv('analysis/outputs/csv/coach_matched_war_background_data.csv', index=False)
    
    print(f"\nDetailed data saved:")
    print("  - coach_backgrounds_from_history.csv: Coach background classifications from actual history")
    print("  - coach_background_trajectories_from_history_15seasons.csv: Average trajectories by background (first 15 seasons)")
    print("  - coach_matched_war_background_data.csv: Individual coach-season data with backgrounds")
    
    plt.close()
    
    return coach_backgrounds, avg_trajectories, matched_data

if __name__ == "__main__":
    print("Analyzing coach backgrounds from actual coaching history...")
    backgrounds, trajectories, matched_data = create_background_war_analysis()
    print(f"\nAnalysis complete! Check the PNG file and CSV outputs for detailed results.")