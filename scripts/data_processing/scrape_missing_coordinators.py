#!/usr/bin/env python3
"""
Scrape Missing Coordinators (DCs and OCs) from PFR Team Pages

The existing coach scraper discovers coaches from PFR's /coaches/ index which
primarily lists head coaches. This script fills gaps by scraping coordinator
information directly from individual team pages.

For each team-year missing a DC or OC in team_rosters.json:
1. Fetches the PFR team page (e.g., /teams/buf/2020.htm)
2. Parses the coaching staff section for DC/OC names and links
3. Scrapes individual coach pages for newly discovered coaches

Usage:
    python scrape_missing_coordinators.py [--start_year 2016] [--end_year 2024]
    python scrape_missing_coordinators.py --dry_run  # Preview what would be scraped
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from random import uniform
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from crawlers.PFR.coach_scraping import CoachDataScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = 'https://www.pro-football-reference.com'

# Mapping from roster key to PFR URL code
# The coaching tree stores each franchise under ALL historical variant codes
# (e.g., Cleveland: cle, cti, cib, cli). This maps every variant to the
# canonical PFR URL code used at pro-football-reference.com/teams/{code}/
ROSTER_KEY_TO_PFR_CODE = {
    # AFC East
    'buf': 'buf', 'bff': 'buf', 'bba': 'buf',
    'mia': 'mia', 'msa': 'mia',
    'nwe': 'nwe',
    'nyj': 'nyj', 'nyt': 'nyj',
    # AFC North
    'rav': 'rav', 'bal': 'rav',
    'cin': 'cin', 'red': 'cin', 'ccl': 'cin',
    'cle': 'cle', 'cti': 'cle', 'cib': 'cle', 'cli': 'cle',
    'pit': 'pit',
    # AFC South — hou/htx/oti are merged in roster data (all Titans)
    'oti': 'oti', 'htx': 'oti', 'hou': 'oti',
    'clt': 'clt', 'ind': 'clt',
    'jax': 'jax',
    # AFC West
    'den': 'den',
    'kan': 'kan', 'kcb': 'kan',
    'rai': 'rai', 'oak': 'rai', 'lvr': 'rai',
    'sdg': 'sdg', 'lac': 'sdg', 'sd': 'sdg',
    # NFC East
    'dal': 'dal',
    'nyg': 'nyg', 'ng1': 'nyg',
    'phi': 'phi',
    'was': 'was', 'sen': 'was',
    # NFC North
    'chi': 'chi',
    'det': 'det', 'dwl': 'det', 'dpn': 'det', 'dhr': 'det', 'dti': 'det',
    'gnb': 'gnb',
    'min': 'min', 'mnn': 'min',
    # NFC South
    'atl': 'atl',
    'car': 'car',
    'nor': 'nor',
    'tam': 'tam',
    # NFC West
    'crd': 'crd', 'ari': 'crd',
    'ram': 'ram', 'lar': 'ram', 'stl': 'ram',
    'sfo': 'sfo',
    'sea': 'sea',
}

# Reverse lookup: PFR code -> all roster keys that map to it
_PFR_TO_ROSTER_KEYS = {}
for _roster_key, _pfr_code in ROSTER_KEY_TO_PFR_CODE.items():
    _PFR_TO_ROSTER_KEYS.setdefault(_pfr_code, []).append(_roster_key)


def load_team_rosters(rosters_path: str) -> Dict:
    """Load team_rosters.json"""
    with open(rosters_path, 'r') as f:
        return json.load(f)


def find_missing_coordinators(rosters: Dict, start_year: int, end_year: int) -> List[Tuple[str, int, str]]:
    """
    Find team-years missing a DC or OC, deduplicated by franchise.

    The roster stores each franchise under multiple historical codes (e.g.,
    Cleveland: cle, cti, cib, cli). We only generate one missing entry per
    franchise per year, using the canonical PFR code.

    Returns:
        List of (pfr_code, year, missing_role) tuples
    """
    missing = []
    seen = set()  # Track (pfr_code, year, role) to avoid duplicates

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        if year_str not in rosters:
            logger.warning(f"Year {year} not found in team_rosters.json")
            continue

        for team, staff in rosters[year_str].items():
            pfr_code = ROSTER_KEY_TO_PFR_CODE.get(team)
            if not pfr_code:
                continue  # Skip unknown historical codes

            for role in ['DC', 'OC']:
                if not staff.get(role):
                    key = (pfr_code, year, role)
                    if key not in seen:
                        seen.add(key)
                        missing.append(key)

    return missing


def get_team_page_url(pfr_code: str, year: int) -> str:
    """Build a PFR team page URL from a canonical PFR code."""
    return f"{BASE_URL}/teams/{pfr_code}/{year}.htm"


def deduplicate_pages(missing: List[Tuple[str, int, str]]) -> Dict[str, List[Tuple[str, int, str]]]:
    """
    Group missing entries by unique team page URL to avoid redundant requests.

    Since find_missing_coordinators already returns canonical PFR codes,
    this just groups by URL (multiple roles for the same team-year).

    Returns:
        Dict mapping URL to list of (pfr_code, year, role) entries
    """
    url_groups = {}
    for pfr_code, year, role in missing:
        url = get_team_page_url(pfr_code, year)
        if url not in url_groups:
            url_groups[url] = []
        url_groups[url].append((pfr_code, year, role))
    return url_groups


def parse_coaching_staff(soup: BeautifulSoup) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Parse coaching staff from a PFR team page.

    Looks for the coaching staff section and extracts coordinator names and URLs.

    Returns:
        Dict mapping role ('DC', 'OC') to (name, coach_url_or_None)
    """
    coordinators = {}

    # PFR team pages list coaching staff in a section, often in the page meta
    # or in a dedicated div. The format varies but typically includes lines like:
    # "Defensive Coordinator: Name" or similar patterns

    # Strategy 1: Look for coaching staff in the page content
    # PFR lists coaching staff in the team page, often in the #meta div
    meta_div = soup.find('div', {'id': 'meta'})
    if meta_div:
        # Look for text containing coordinator info
        for p_tag in meta_div.find_all('p'):
            text = p_tag.get_text(strip=True)

            # Match patterns like "Defensive Coord.: Name" or "Off. Coordinator: Name"
            dc_patterns = [
                r'(?:Defensive\s+Coord(?:inator)?\.?\s*[:\-])\s*(.+)',
                r'(?:Def\.\s*Coord(?:inator)?\.?\s*[:\-])\s*(.+)',
            ]
            oc_patterns = [
                r'(?:Offensive\s+Coord(?:inator)?\.?\s*[:\-])\s*(.+)',
                r'(?:Off\.\s*Coord(?:inator)?\.?\s*[:\-])\s*(.+)',
            ]

            for pattern in dc_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and 'DC' not in coordinators:
                    name = match.group(1).strip()
                    # Try to find a link for this coach
                    link = _find_coach_link(p_tag, name)
                    coordinators['DC'] = (name, link)

            for pattern in oc_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and 'OC' not in coordinators:
                    name = match.group(1).strip()
                    link = _find_coach_link(p_tag, name)
                    coordinators['OC'] = (name, link)

    # Strategy 2: Look in any coaching staff table or list on the page
    # Some PFR pages have a "Coaching Staff" section
    for heading in soup.find_all(['h2', 'h3', 'strong']):
        if 'coach' in heading.get_text(strip=True).lower():
            # Look in subsequent elements for coordinator info
            sibling = heading.find_next_sibling()
            if sibling:
                _parse_staff_element(sibling, coordinators)

    return coordinators


def _find_coach_link(element, coach_name: str) -> Optional[str]:
    """Find the PFR URL for a coach within an HTML element."""
    for link in element.find_all('a'):
        href = link.get('href', '')
        if href.startswith('/coaches/') and href.endswith('.htm'):
            # Check that the link text roughly matches the coach name
            link_text = link.get_text(strip=True)
            if link_text and (link_text in coach_name or coach_name in link_text):
                return href
    return None


def _parse_staff_element(element, coordinators: Dict):
    """Parse a coaching staff HTML element for coordinator info."""
    text = element.get_text(separator='\n')
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        dc_match = re.search(r'(?:Defensive\s+Coord|DC)\s*[:\-]\s*(.+)', line, re.IGNORECASE)
        if dc_match and 'DC' not in coordinators:
            name = dc_match.group(1).strip()
            link = _find_coach_link(element, name)
            coordinators['DC'] = (name, link)

        oc_match = re.search(r'(?:Offensive\s+Coord|OC)\s*[:\-]\s*(.+)', line, re.IGNORECASE)
        if oc_match and 'OC' not in coordinators:
            name = oc_match.group(1).strip()
            link = _find_coach_link(element, name)
            coordinators['OC'] = (name, link)


def rate_limit():
    """Apply rate limiting between PFR requests (3-6 seconds)."""
    delay = uniform(3.0, 6.0)
    time.sleep(delay)


def scrape_team_pages(url_groups: Dict, dry_run: bool = False) -> Tuple[Dict, Set[str]]:
    """
    Scrape team pages and extract coordinator info.

    Args:
        url_groups: Mapping of URL to list of (team_code, year, role) entries
        dry_run: If True, don't actually make requests

    Returns:
        Tuple of (found_coordinators, new_coach_urls_to_scrape)
        found_coordinators: {(team_code, year, role): (name, url)}
        new_coach_urls_to_scrape: set of coach URLs not yet scraped
    """
    found = {}
    new_coach_urls = set()
    total = len(url_groups)

    existing_coaches = _get_existing_coach_dirs()

    for i, (url, entries) in enumerate(url_groups.items(), 1):
        team_code = entries[0][0]
        year = entries[0][1]
        roles_needed = [e[2] for e in entries]

        logger.info(f"Scraping {team_code} {year} (need {', '.join(roles_needed)})... ({i}/{total})")

        if dry_run:
            continue

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')
            coordinators = parse_coaching_staff(soup)

            for entry in entries:
                tc, yr, role = entry
                if role in coordinators:
                    name, coach_url = coordinators[role]
                    found[(tc, yr, role)] = (name, coach_url)
                    logger.info(f"  Found {role}: {name}")

                    # Track new coach URLs that need scraping
                    if coach_url:
                        # Check if coach already scraped
                        coach_page_name = _url_to_coach_name(coach_url)
                        if coach_page_name and coach_page_name not in existing_coaches:
                            new_coach_urls.add(coach_url)
                else:
                    logger.info(f"  No {role} found on page (may genuinely lack one)")

            rate_limit()

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            rate_limit()
            continue

    return found, new_coach_urls


def _get_existing_coach_dirs() -> Set[str]:
    """Get set of already-scraped coach directory names."""
    coaches_dir = Path("data/raw/Coaches")
    if coaches_dir.exists():
        return {d.name for d in coaches_dir.iterdir() if d.is_dir()}
    return set()


def _url_to_coach_name(coach_url: str) -> Optional[str]:
    """Convert a coach URL to the expected directory name (we can't know for sure, just the URL ID)."""
    # Coach URLs look like /coaches/RyanRe0.htm
    # The actual directory name is the coach's full name which we get after scraping
    # Return the URL ID as a proxy for tracking
    if coach_url:
        return Path(coach_url).stem
    return None


def scrape_new_coaches(coach_urls: Set[str], coaches_dir: str = "data/raw/Coaches",
                       dry_run: bool = False) -> Tuple[int, int]:
    """
    Scrape individual coach pages for newly discovered coaches.

    Args:
        coach_urls: Set of coach page URLs to scrape
        coaches_dir: Directory to save coach data
        dry_run: If True, just report what would be scraped

    Returns:
        Tuple of (successful, failed) counts
    """
    if not coach_urls:
        logger.info("No new coaches to scrape")
        return 0, 0

    logger.info(f"Scraping {len(coach_urls)} new coach pages...")

    if dry_run:
        for url in sorted(coach_urls):
            logger.info(f"  Would scrape: {url}")
        return 0, 0

    scraper = CoachDataScraper(output_dir=coaches_dir)
    successful = 0
    failed = 0

    for i, coach_url in enumerate(sorted(coach_urls), 1):
        logger.info(f"Scraping coach {i}/{len(coach_urls)}: {coach_url}")

        if scraper.scrape_coach_data(coach_url):
            successful += 1
        else:
            failed += 1

        rate_limit()

    return successful, failed


def update_team_rosters(rosters_path: str, found: Dict) -> int:
    """
    Update team_rosters.json with newly found coordinators.

    Args:
        rosters_path: Path to team_rosters.json
        found: Dict of {(team_code, year, role): (name, url)}

    Returns:
        Number of entries updated
    """
    if not found:
        return 0

    with open(rosters_path, 'r') as f:
        rosters = json.load(f)

    updated = 0
    # We also need coaches.json to map names to coach_ids
    coaches_path = Path(rosters_path).parent / "coaches.json"
    coach_id_lookup = {}

    if coaches_path.exists():
        with open(coaches_path, 'r') as f:
            coaches = json.load(f)
        # Build name-to-id lookup
        for coach_id, coach_data in coaches.items():
            name = coach_data.get('name', '')
            coach_id_lookup[name] = coach_id

    for (pfr_code, year, role), (name, url) in found.items():
        year_str = str(year)
        if year_str not in rosters:
            continue

        # Try to find coach_id from coaches.json
        coach_id = coach_id_lookup.get(name)

        # If not found by exact name, try the URL stem
        if not coach_id and url:
            url_stem = Path(url).stem
            if url_stem in coaches:
                coach_id = url_stem

        if coach_id:
            # Update ALL roster variant codes for this franchise
            variant_keys = _PFR_TO_ROSTER_KEYS.get(pfr_code, [pfr_code])
            for roster_key in variant_keys:
                if roster_key in rosters[year_str]:
                    rosters[year_str][roster_key][role] = coach_id
            updated += 1
            logger.info(f"Updated {pfr_code} {year} {role} = {coach_id} ({name})")
        else:
            logger.warning(f"Could not find coach_id for {name} ({url}) - "
                         f"rebuild coaching tree after scraping to resolve")

    if updated > 0:
        with open(rosters_path, 'w') as f:
            json.dump(rosters, f, indent=2)
        logger.info(f"Saved updated team_rosters.json with {updated} new entries")

    return updated


def print_summary(missing: List, found: Dict, new_urls: Set, coach_success: int,
                  coach_failed: int, updated: int):
    """Print a summary of what was done."""
    print("\n" + "=" * 80)
    print("COORDINATOR SCRAPING SUMMARY")
    print("=" * 80)

    # Count by role
    missing_dc = sum(1 for _, _, r in missing if r == 'DC')
    missing_oc = sum(1 for _, _, r in missing if r == 'OC')
    found_dc = sum(1 for (_, _, r) in found if r == 'DC')
    found_oc = sum(1 for (_, _, r) in found if r == 'OC')

    print(f"\nMissing coordinators identified:")
    print(f"  DC: {missing_dc} team-years")
    print(f"  OC: {missing_oc} team-years")
    print(f"  Total: {len(missing)} team-years")

    print(f"\nCoordinators found on team pages:")
    print(f"  DC: {found_dc}")
    print(f"  OC: {found_oc}")
    print(f"  Total: {len(found)}")

    print(f"\nNew coach pages scraped: {coach_success} successful, {coach_failed} failed")
    print(f"Roster entries updated: {updated}")
    print(f"\nNote: Run build_coaching_tree.py to rebuild with new coordinator data")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Scrape missing DCs and OCs from PFR team pages')
    parser.add_argument('--start_year', type=int, default=2016,
                        help='Start year (default: 2016, when defensive tracking data begins)')
    parser.add_argument('--end_year', type=int, default=2024,
                        help='End year (default: 2024)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview what would be scraped without making requests')
    parser.add_argument('--rosters_path', type=str,
                        default='data/processed/coaching_tree/team_rosters.json',
                        help='Path to team_rosters.json')
    parser.add_argument('--coaches_dir', type=str, default='data/raw/Coaches',
                        help='Directory for coach data')

    args = parser.parse_args()

    print("=" * 80)
    print("SCRAPE MISSING COORDINATORS")
    print("=" * 80)
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80 + "\n")

    # Step 1: Load rosters and find gaps
    logger.info("Step 1: Loading team rosters and identifying gaps...")
    rosters = load_team_rosters(args.rosters_path)
    missing = find_missing_coordinators(rosters, args.start_year, args.end_year)
    logger.info(f"Found {len(missing)} missing coordinator entries")

    if not missing:
        print("No missing coordinators found!")
        return

    # Step 2: Deduplicate by URL
    logger.info("Step 2: Deduplicating team page URLs...")
    url_groups = deduplicate_pages(missing)
    logger.info(f"Need to scrape {len(url_groups)} unique team pages")

    # Step 3: Scrape team pages
    logger.info("Step 3: Scraping team pages for coordinator info...")
    found, new_coach_urls = scrape_team_pages(url_groups, dry_run=args.dry_run)

    # Step 4: Scrape new coach pages
    logger.info("Step 4: Scraping individual coach pages...")
    coach_success, coach_failed = scrape_new_coaches(
        new_coach_urls, args.coaches_dir, dry_run=args.dry_run)

    # Step 5: Update team_rosters.json
    updated = 0
    if not args.dry_run and found:
        logger.info("Step 5: Updating team_rosters.json...")
        updated = update_team_rosters(args.rosters_path, found)

    print_summary(missing, found, new_coach_urls, coach_success, coach_failed, updated)


if __name__ == "__main__":
    main()
