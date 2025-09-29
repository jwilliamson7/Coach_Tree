#!/usr/bin/env python3
"""
Quick test script to verify no_play classification logic works
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Read a sample of play-by-play data
data_dir = Path("data/raw/play_by_play")
sample_file = list(data_dir.glob("play_by_play_2023.csv"))[0]

print("Loading sample data from 2023...")
df = pd.read_csv(sample_file)

# Filter for no_play plays
no_play_df = df[df['play_type'] == 'no_play'].copy()
print(f"Found {len(no_play_df):,} no_play plays in 2023")

# Sample some no_play descriptions
print("\nSample no_play descriptions:")
print("-" * 80)
sample_descs = no_play_df[['desc']].head(20)
for idx, row in sample_descs.iterrows():
    print(f"{row['desc'][:150]}...")

# Apply classification logic
print("\n" + "=" * 80)
print("Testing classification logic on no_play plays")
print("=" * 80)

# Define pass play keywords
pass_keywords = [
    ' pass ', ' passes ', ' passed ', ' passing ',
    'incomplete', 'complete pass', 'sacked', ' sack ',
    'scrambles', 'scrambled', 'throw', 'throws',
    'intercepted', 'interception'
]

# Define run play keywords
run_keywords = [
    ' run ', ' runs ', ' ran ', ' rush ', ' rushes ', ' rushed ',
    'up the middle', 'left end', 'right end', 
    'left tackle', 'right tackle', 'left guard', 'right guard'
]

# Convert descriptions to lowercase
desc_lower = no_play_df['desc'].str.lower()

# Check for pass keywords
pass_pattern = '|'.join(pass_keywords)
pass_matches = desc_lower.str.contains(pass_pattern, na=False, regex=True)

# Check for run keywords
run_pattern = '|'.join(run_keywords)
run_matches = desc_lower.str.contains(run_pattern, na=False, regex=True)

# Count classifications
pass_count = pass_matches.sum()
run_count = (run_matches & ~pass_matches).sum()
unclassified = len(no_play_df) - pass_count - run_count

print(f"\nClassification results:")
print(f"  Pass plays: {pass_count:,} ({pass_count/len(no_play_df)*100:.1f}%)")
print(f"  Run plays: {run_count:,} ({run_count/len(no_play_df)*100:.1f}%)")
print(f"  Unclassified: {unclassified:,} ({unclassified/len(no_play_df)*100:.1f}%)")

# Show examples of classified plays
print("\n" + "=" * 80)
print("Examples of classified pass plays:")
print("-" * 80)
pass_examples = no_play_df[pass_matches]['desc'].head(5)
for desc in pass_examples:
    print(f"  {desc[:150]}...")

print("\n" + "=" * 80)
print("Examples of classified run plays:")
print("-" * 80)
run_examples = no_play_df[run_matches & ~pass_matches]['desc'].head(5)
for desc in run_examples:
    print(f"  {desc[:150]}...")

print("\n" + "=" * 80)
print("Examples of unclassified plays (likely special teams or procedural):")
print("-" * 80)
unclassified_examples = no_play_df[~pass_matches & ~run_matches]['desc'].head(5)
for desc in unclassified_examples:
    print(f"  {desc[:150]}...")