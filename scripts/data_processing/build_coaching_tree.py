import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add project root to path to import from crawlers
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crawlers.utils.data_constants import (
    TEAM_FRANCHISE_MAPPINGS,
    EXCLUDED_ROLE_KEYWORDS,
    standardize_team_abbreviation,
    FULL_TEAM_NAME_TO_PFR_ABBREV
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CoachingPosition:
    """Represents a single coaching position in a given year"""
    year: int
    team: str
    team_franchises: List[str]  # All franchise variants for this team
    level: str  # NFL, College, or None
    role: str  # Head Coach, Offensive Coordinator, etc.
    role_category: str  # HC, OC, DC, STC, Position, None
    age: Optional[int] = None
    employer: Optional[str] = None
    parent_hc_id: Optional[str] = None  # ID of parent head coach
    parent_coordinator_id: Optional[str] = None  # ID of parent coordinator (for position coaches)


class Coach:
    """Represents a coach with their entire career timeline"""
    
    def __init__(self, coach_name: str, coach_id: str, history_file: Path):
        """Initialize coach with career history"""
        self.name = coach_name
        self.id = coach_id
        self.history_file = history_file
        self.career_timeline: Dict[int, CoachingPosition] = {}
        self.load_career_history()
    
    def _classify_coaching_role(self, role: str) -> Tuple[str, str]:
        """
        Classify coaching role and category
        Returns: (cleaned_role, role_category)
        """
        if not role or not isinstance(role, str):
            return ("None", "None")
        
        # Check for excluded keywords (interim, etc.)
        for keyword in EXCLUDED_ROLE_KEYWORDS:
            if keyword.lower() in role.lower():
                return ("None", "None")
        
        # Exclude generic assistant roles without specific position
        if ("Assistant" in role or "Asst" in role) and "/" not in role and "\\" not in role:
            if "Head Coach" not in role and "Coordinator" not in role:
                return ("None", "None")
        
        # Classify specific roles
        if "Head Coach" in role and "Ass" not in role and "Interim" not in role:
            return ("Head Coach", "HC")
        
        if "Coordinator" in role:
            if "Offensive Coordinator" in role and "Interim O" not in role:
                return ("Offensive Coordinator", "OC")
            elif "Defensive Coordinator" in role and "Interim D" not in role:
                return ("Defensive Coordinator", "DC")
            elif "Special" in role and "Interim S" not in role:
                return ("Special Teams Coordinator", "STC")
        
        # Position coach classification
        # Offensive positions
        offensive_keywords = ['Quarterback', 'QB', 'Running Back', 'RB', 'Offensive Line', 
                             'OL', 'Wide Receiver', 'WR', 'Tight End', 'TE', 'Pass', 'Run Game']
        # Defensive positions  
        defensive_keywords = ['Defensive', 'DB', 'Linebacker', 'LB', 'Defensive Line', 
                             'DL', 'Secondary', 'Cornerback', 'CB', 'Safety', 'Pass Rush']
        # Special teams positions
        special_keywords = ['Special Teams', 'Kicking', 'Punting', 'Return']
        
        role_upper = role.upper()
        
        if any(kw.upper() in role_upper for kw in offensive_keywords):
            return (role, "Position_Offensive")
        elif any(kw.upper() in role_upper for kw in defensive_keywords):
            return (role, "Position_Defensive")
        elif any(kw.upper() in role_upper for kw in special_keywords):
            return (role, "Position_Special")
        
        # Default position coach
        return (role, "Position")
    
    def _classify_coaching_level(self, level: str) -> str:
        """Classify coaching level (College, NFL, or None)"""
        if not level:
            return "None"
        if "College" in level:
            return "College"
        if "NFL" in level and level != "NFL Europe":
            return "NFL"
        return "None"
    
    def _get_team_abbreviation(self, team_name: str, year: int = None) -> str:
        """Extract team abbreviation from team name using standardized mapping.

        Uses the centralized standardize_team_abbreviation function from data_constants.py
        which handles full team names, abbreviations, and year-based franchise logic.

        Note: This mapping is optimized for the 2006+ play-by-play data era.
        Pre-2006 team abbreviations may not be fully accurate for historical analysis.
        """
        if not team_name:
            return ""

        return standardize_team_abbreviation(team_name, year)
    
    def _resolve_team_franchises(self, team_abbrev: str) -> List[str]:
        """Resolve team abbreviation to all historical franchise variants"""
        if not team_abbrev:
            return []
        
        # Check if this abbreviation is in the mappings
        if team_abbrev in TEAM_FRANCHISE_MAPPINGS:
            franchise_list = TEAM_FRANCHISE_MAPPINGS[team_abbrev]
            return franchise_list if isinstance(franchise_list, list) else [franchise_list]
        
        # Check if this is a variant of another team
        for main_abbrev, variants in TEAM_FRANCHISE_MAPPINGS.items():
            if isinstance(variants, list) and team_abbrev in variants:
                return variants
            elif team_abbrev == variants:
                return [main_abbrev, variants]
        
        # Return as-is if not found
        return [team_abbrev]
    
    def load_career_history(self):
        """Load and parse coaching career history from CSV"""
        if not self.history_file.exists():
            logger.warning(f"History file not found for {self.name}: {self.history_file}")
            return
        
        try:
            df = pd.read_csv(self.history_file)
            
            current_year = None
            for _, row in df.iterrows():
                year = int(row['Year'])
                
                # Skip if we've already processed this year (use first occurrence only)
                if year in self.career_timeline:
                    continue
                
                level = self._classify_coaching_level(row.get('Level', ''))
                role_raw = row.get('Role', '')
                role, role_category = self._classify_coaching_role(role_raw)
                
                # Skip if role is None
                if role == "None":
                    continue
                
                # Get team information
                team_raw = row.get('Tm', '') or row.get('Team', '') or row.get('Employer', '')
                team_abbrev = self._get_team_abbreviation(team_raw, year)
                # Use the single year-aware canonical franchise key for identity
                # (consistent with Coach_WAR / CoachingProject). Expanding to the
                # full variant list collapsed distinct franchises that share a
                # legacy code (Texans/Titans 'oti', Colts/Ravens 'clt'), producing
                # false cross-franchise mentor-protege edges. The canonical key
                # already merges genuine relocations (OAK/LV->rai, SD/LAC->sdg,
                # STL/LAR->ram) while keeping distinct franchises distinct.
                team_franchises = [team_abbrev] if team_abbrev else []
                
                # Create position record
                position = CoachingPosition(
                    year=year,
                    team=team_abbrev,
                    team_franchises=team_franchises,
                    level=level,
                    role=role,
                    role_category=role_category,
                    age=row.get('Age') if pd.notna(row.get('Age')) else None,
                    employer=row.get('Employer', '')
                )
                
                self.career_timeline[year] = position
                
        except Exception as e:
            logger.error(f"Error loading history for {self.name}: {e}")
    
    def get_position(self, year: int) -> Optional[CoachingPosition]:
        """Get coaching position for a specific year"""
        return self.career_timeline.get(year)
    
    def get_nfl_years(self) -> List[int]:
        """Get all years when coach was in NFL"""
        return [year for year, pos in self.career_timeline.items() 
                if pos.level == "NFL"]
    
    def to_dict(self) -> Dict:
        """Convert coach to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'career': {
                year: {
                    'team': pos.team,
                    'team_franchises': pos.team_franchises,
                    'level': pos.level,
                    'role': pos.role,
                    'role_category': pos.role_category,
                    'age': pos.age,
                    'parent_hc_id': pos.parent_hc_id,
                    'parent_coordinator_id': pos.parent_coordinator_id
                }
                for year, pos in self.career_timeline.items()
            }
        }


class CoachingTree:
    """Manages the complete coaching tree with all relationships"""
    
    def __init__(self, coaches_dir: str = "data/raw/Coaches"):
        """Initialize the coaching tree"""
        self.coaches_dir = Path(coaches_dir)
        self.coaches: Dict[str, Coach] = {}
        self.team_rosters: Dict[int, Dict[str, Dict[str, str]]] = {}  # year -> team -> role -> coach_id
        self.relationships: List[Dict] = []  # List of parent-child relationships
        
    def load_all_coaches(self):
        """Load all coaches from the data directory"""
        coach_dirs = [d for d in self.coaches_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Loading {len(coach_dirs)} coaches...")
        
        for coach_dir in coach_dirs:
            coach_name = coach_dir.name
            
            # Skip if visited.txt or other non-coach directories
            if coach_name in ['visited.txt', 'visited_teams.txt']:
                continue
            
            # Generate unique coach ID (using name for simplicity)
            coach_id = coach_name.replace(' ', '_').lower()
            
            # Load coaching history
            history_file = coach_dir / "all_coaching_history.csv"
            if history_file.exists():
                coach = Coach(coach_name, coach_id, history_file)
                self.coaches[coach_id] = coach
            else:
                logger.debug(f"No history file for {coach_name}")
        
        logger.info(f"Loaded {len(self.coaches)} coaches successfully")
    
    def build_team_rosters(self):
        """Build team rosters for each year to enable cross-referencing"""
        logger.info("Building team rosters by year...")
        
        for coach_id, coach in self.coaches.items():
            for year, position in coach.career_timeline.items():
                if position.level != "NFL":
                    continue
                
                # Initialize year if needed
                if year not in self.team_rosters:
                    self.team_rosters[year] = {}
                
                # Add coach to all franchise variants
                for team in position.team_franchises:
                    if team not in self.team_rosters[year]:
                        self.team_rosters[year][team] = {}
                    
                    # Store by role category for easier lookup
                    role_key = position.role_category
                    if role_key == "HC":
                        self.team_rosters[year][team]["HC"] = coach_id
                    elif role_key == "OC":
                        self.team_rosters[year][team]["OC"] = coach_id
                    elif role_key == "DC":
                        self.team_rosters[year][team]["DC"] = coach_id
                    elif role_key == "STC":
                        self.team_rosters[year][team]["STC"] = coach_id
                    else:
                        # For position coaches, store in a list
                        if "Position_Coaches" not in self.team_rosters[year][team]:
                            self.team_rosters[year][team]["Position_Coaches"] = []
                        self.team_rosters[year][team]["Position_Coaches"].append({
                            'coach_id': coach_id,
                            'role': position.role,
                            'category': position.role_category
                        })
        
        logger.info(f"Built rosters for {len(self.team_rosters)} years")
    
    def assign_parent_relationships(self):
        """Assign parent relationships based on team rosters"""
        logger.info("Assigning parent-child relationships...")
        
        for coach_id, coach in self.coaches.items():
            for year, position in coach.career_timeline.items():
                if position.level != "NFL":
                    continue
                
                # Find the team's roster for this year
                if year not in self.team_rosters:
                    continue
                
                team_roster = None
                for team in position.team_franchises:
                    if team in self.team_rosters[year]:
                        team_roster = self.team_rosters[year][team]
                        break
                
                if not team_roster:
                    continue
                
                # Assign parent relationships based on role
                if position.role_category in ["OC", "DC", "STC"]:
                    # Coordinators report to head coach
                    if "HC" in team_roster:
                        position.parent_hc_id = team_roster["HC"]
                        self.relationships.append({
                            'year': year,
                            'child_id': coach_id,
                            'child_name': coach.name,
                            'child_role': position.role,
                            'parent_id': team_roster["HC"],
                            'parent_name': self.coaches[team_roster["HC"]].name,
                            'parent_role': 'Head Coach',
                            'team': position.team,
                            'relationship_type': 'coordinator_to_hc'
                        })
                
                elif position.role_category.startswith("Position"):
                    # Position coaches report to coordinators and head coach
                    
                    # Find parent coordinator
                    if position.role_category == "Position_Offensive" and "OC" in team_roster:
                        position.parent_coordinator_id = team_roster["OC"]
                        self.relationships.append({
                            'year': year,
                            'child_id': coach_id,
                            'child_name': coach.name,
                            'child_role': position.role,
                            'parent_id': team_roster["OC"],
                            'parent_name': self.coaches[team_roster["OC"]].name,
                            'parent_role': 'Offensive Coordinator',
                            'team': position.team,
                            'relationship_type': 'position_to_coordinator'
                        })
                    elif position.role_category == "Position_Defensive" and "DC" in team_roster:
                        position.parent_coordinator_id = team_roster["DC"]
                        self.relationships.append({
                            'year': year,
                            'child_id': coach_id,
                            'child_name': coach.name,
                            'child_role': position.role,
                            'parent_id': team_roster["DC"],
                            'parent_name': self.coaches[team_roster["DC"]].name,
                            'parent_role': 'Defensive Coordinator',
                            'team': position.team,
                            'relationship_type': 'position_to_coordinator'
                        })
                    elif position.role_category == "Position_Special" and "STC" in team_roster:
                        position.parent_coordinator_id = team_roster["STC"]
                        self.relationships.append({
                            'year': year,
                            'child_id': coach_id,
                            'child_name': coach.name,
                            'child_role': position.role,
                            'parent_id': team_roster["STC"],
                            'parent_name': self.coaches[team_roster["STC"]].name,
                            'parent_role': 'Special Teams Coordinator',
                            'team': position.team,
                            'relationship_type': 'position_to_coordinator'
                        })
                    
                    # All position coaches also report to head coach
                    if "HC" in team_roster:
                        position.parent_hc_id = team_roster["HC"]
                        self.relationships.append({
                            'year': year,
                            'child_id': coach_id,
                            'child_name': coach.name,
                            'child_role': position.role,
                            'parent_id': team_roster["HC"],
                            'parent_name': self.coaches[team_roster["HC"]].name,
                            'parent_role': 'Head Coach',
                            'team': position.team,
                            'relationship_type': 'position_to_hc'
                        })
        
        logger.info(f"Created {len(self.relationships)} relationships")
    
    def get_coaching_lineage(self, coach_id: str) -> Dict:
        """Get the complete coaching lineage for a specific coach"""
        if coach_id not in self.coaches:
            return {}
        
        coach = self.coaches[coach_id]
        lineage = {
            'coach': coach.name,
            'id': coach_id,
            'mentors': [],  # Coaches who were parents
            'proteges': []  # Coaches who were children
        }
        
        # Find all mentors (where this coach was child)
        mentor_ids = set()
        for rel in self.relationships:
            if rel['child_id'] == coach_id:
                if rel['parent_id'] not in mentor_ids:
                    mentor_ids.add(rel['parent_id'])
                    lineage['mentors'].append({
                        'id': rel['parent_id'],
                        'name': rel['parent_name'],
                        'year': rel['year'],
                        'role': rel['parent_role'],
                        'team': rel['team']
                    })
        
        # Find all proteges (where this coach was parent)
        protege_ids = set()
        for rel in self.relationships:
            if rel['parent_id'] == coach_id:
                if rel['child_id'] not in protege_ids:
                    protege_ids.add(rel['child_id'])
                    lineage['proteges'].append({
                        'id': rel['child_id'],
                        'name': rel['child_name'],
                        'year': rel['year'],
                        'role': rel['child_role'],
                        'team': rel['team']
                    })
        
        return lineage
    
    def save_tree(self, output_dir: str = "data/processed/coaching_tree"):
        """Save the coaching tree to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save all coaches with their timelines
        coaches_data = {
            coach_id: coach.to_dict() 
            for coach_id, coach in self.coaches.items()
        }
        
        with open(output_path / "coaches.json", 'w') as f:
            json.dump(coaches_data, f, indent=2)
        
        # Save relationships as CSV for easier analysis
        relationships_df = pd.DataFrame(self.relationships)
        relationships_df.to_csv(output_path / "relationships.csv", index=False)
        
        # Save team rosters
        with open(output_path / "team_rosters.json", 'w') as f:
            json.dump(self.team_rosters, f, indent=2)
        
        logger.info(f"Saved coaching tree to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the coaching tree"""
        stats = {
            'total_coaches': len(self.coaches),
            'total_relationships': len(self.relationships),
            'years_covered': sorted(self.team_rosters.keys()) if self.team_rosters else [],
            'coaches_by_role': {},
            'relationships_by_type': {}
        }
        
        # Count coaches by their highest role achieved
        for coach in self.coaches.values():
            highest_role = "None"
            for position in coach.career_timeline.values():
                if position.role_category == "HC":
                    highest_role = "Head Coach"
                    break
                elif position.role_category in ["OC", "DC", "STC"] and highest_role not in ["Head Coach"]:
                    highest_role = "Coordinator"
                elif position.role_category.startswith("Position") and highest_role == "None":
                    highest_role = "Position Coach"
            
            stats['coaches_by_role'][highest_role] = stats['coaches_by_role'].get(highest_role, 0) + 1
        
        # Count relationships by type
        for rel in self.relationships:
            rel_type = rel['relationship_type']
            stats['relationships_by_type'][rel_type] = stats['relationships_by_type'].get(rel_type, 0) + 1
        
        return stats


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("NFL COACHING TREE BUILDER")
    print("="*60 + "\n")
    
    # Initialize the coaching tree
    tree = CoachingTree()
    
    # Load all coaches
    print("Step 1: Loading all coaches...")
    tree.load_all_coaches()
    
    # Build team rosters
    print("\nStep 2: Building team rosters by year...")
    tree.build_team_rosters()
    
    # Assign parent relationships
    print("\nStep 3: Assigning parent-child relationships...")
    tree.assign_parent_relationships()
    
    # Save the tree
    print("\nStep 4: Saving coaching tree data...")
    tree.save_tree()
    
    # Display statistics
    print("\n" + "="*60)
    print("COACHING TREE STATISTICS")
    print("="*60)
    
    stats = tree.get_statistics()
    print(f"\nTotal Coaches: {stats['total_coaches']}")
    print(f"Total Relationships: {stats['total_relationships']}")
    print(f"Years Covered: {min(stats['years_covered'])} - {max(stats['years_covered'])} ({len(stats['years_covered'])} years)")
    
    print("\nCoaches by Highest Role Achieved:")
    for role, count in sorted(stats['coaches_by_role'].items()):
        print(f"  {role}: {count}")
    
    print("\nRelationships by Type:")
    for rel_type, count in sorted(stats['relationships_by_type'].items()):
        print(f"  {rel_type}: {count}")
    
    # Example: Show lineage for a specific coach
    print("\n" + "="*60)
    print("EXAMPLE COACHING LINEAGE")
    print("="*60)
    
    # Find a coach with interesting lineage (e.g., Bill Belichick)
    example_coaches = ['Bill Belichick', 'Andy Reid', 'Mike Tomlin', 'Sean McVay']
    for coach_name in example_coaches:
        coach_id = coach_name.replace(' ', '_').lower()
        if coach_id in tree.coaches:
            print(f"\nLineage for {coach_name}:")
            lineage = tree.get_coaching_lineage(coach_id)
            print(f"  Mentors: {len(lineage['mentors'])}")
            for mentor in lineage['mentors'][:3]:  # Show first 3
                print(f"    - {mentor['name']} ({mentor['role']}, {mentor['year']})")
            print(f"  Proteges: {len(lineage['proteges'])}")
            for protege in lineage['proteges'][:3]:  # Show first 3
                print(f"    - {protege['name']} ({protege['role']}, {protege['year']})")
            break
    
    print("\n" + "="*60)
    print("COACHING TREE BUILD COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()