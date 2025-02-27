#!/usr/bin/env python3
import requests
import re
import json
import os
from datetime import datetime, timedelta, timezone
import argparse
import calendar

def parse_month_name(month_str):
    """Convert month name to month number"""
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    return month_map.get(month_str.lower(), None)

def parse_relative_time(time_text, current_date=None):
    """Parse relative time expressions like '4 days ago', 'last week', etc."""
    if not current_date:
        current_date = datetime.now()
    
    if not time_text:
        return None
        
    time_text = time_text.strip().lower()
    
    # Handle "X hours/days/weeks/months/years ago"
    ago_match = re.search(r'(\d+)\s+(hour|day|week|month|year)s?\s+ago', time_text)
    if ago_match:
        value = int(ago_match.group(1))
        unit = ago_match.group(2)
        
        if unit == 'hour':
            return current_date - timedelta(hours=value)
        elif unit == 'day':
            return current_date - timedelta(days=value)
        elif unit == 'week':
            return current_date - timedelta(days=7*value)
        elif unit == 'month':
            # More accurate month calculation
            year = current_date.year
            month = current_date.month - value
            
            # Handle month rollover
            while month <= 0:
                month += 12
                year -= 1
                
            # Use the same day, but cap at the max days in the resulting month
            day = min(current_date.day, calendar.monthrange(year, month)[1])
            return datetime(year, month, day)
        elif unit == 'year':
            # More accurate year calculation
            year = current_date.year - value
            month = current_date.month
            
            # Use the same day, but handle leap year issues
            day = min(current_date.day, calendar.monthrange(year, month)[1])
            return datetime(year, month, day)
    
    # Handle specific cases
    if 'yesterday' in time_text:
        return current_date - timedelta(days=1)
    elif 'last week' in time_text:
        return current_date - timedelta(days=7)
    elif 'last month' in time_text:
        # Go back exactly one month
        month = current_date.month - 1
        year = current_date.year
        if month == 0:
            month = 12
            year -= 1
        day = min(current_date.day, calendar.monthrange(year, month)[1])
        return datetime(year, month, day)
    elif 'last year' in time_text:
        return datetime(current_date.year - 1, current_date.month, 
                        min(current_date.day, calendar.monthrange(current_date.year - 1, current_date.month)[1]))
    
    return None

def parse_date_text(date_text, current_date=None):
    """Parse date text in various formats"""
    if not current_date:
        current_date = datetime.now()
    
    if not date_text:
        return None
        
    date_text = date_text.strip()
    
    # Try relative time parsing first
    relative_date = parse_relative_time(date_text, current_date)
    if relative_date:
        return relative_date
    
    # Format: "Oct 22, 2024" or "October 22, 2024" or "Oct 22" (current year implied)
    month_day_year_match = re.search(r'([A-Za-z]{3,9})\s+(\d{1,2})(?:[,\s]+(\d{4}))?', date_text)
    if month_day_year_match:
        month_str = month_day_year_match.group(1)
        day = int(month_day_year_match.group(2))
        year = int(month_day_year_match.group(3)) if month_day_year_match.group(3) else current_date.year
        
        month_num = parse_month_name(month_str)
        if month_num:
            # If no year provided and the resulting date would be in the future, use previous year
            if not month_day_year_match.group(3):
                try:
                    result_date = datetime(year, month_num, day)
                    if result_date > current_date:
                        year -= 1
                except ValueError:
                    # Handle invalid dates like Feb 30
                    pass
            
            try:
                return datetime(year, month_num, min(day, calendar.monthrange(year, month_num)[1]))
            except ValueError:
                # Handle invalid dates like Feb 30
                return None
    
    # Format: "2024-10-22" or "2024/10/22"
    iso_date_match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_text)
    if iso_date_match:
        year = int(iso_date_match.group(1))
        month = int(iso_date_match.group(2))
        day = int(iso_date_match.group(3))
        
        try:
            return datetime(year, month, min(day, calendar.monthrange(year, month)[1]))
        except ValueError:
            return None
            
    return None

def extract_github_releases(repo_url, months=6, output_file=None, debug=False, raw_html=False, rate_limit=False):
    """
    Extract GitHub release data with improved date recognition
    
    Args:
        repo_url (str): GitHub repository URL
        months (int): Number of months to look back
        output_file (str): Output file path
        debug (bool): Enable additional debug output
        raw_html (bool): Save raw HTML for debugging
        rate_limit (bool): Add delay to avoid rate limiting
    """
    # Format URL correctly
    if not repo_url.startswith('http'):
        repo_url = f"https://{repo_url}"
    
    base_url = repo_url
    if not base_url.endswith('/releases'):
        if base_url.endswith('/'):
            base_url = f"{base_url}releases"
        else:
            base_url = f"{base_url}/releases"
    
    print(f"Accessing base URL: {base_url}")
    
    # Check if this is an Avalanche repository
    is_avalanche = 'ava-labs/avalanchego' in repo_url
    is_optimism = 'ethereum-optimism/optimism' in repo_url
    is_succinct = 'succinctlabs/op-succinct' in repo_url
    
    # Set up date range
    today = datetime.now()
    lookback_date = today - timedelta(days=30 * months)
    
    # Make the request headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Extract repository name
    repo_name = repo_url.split('github.com/')[1].split('/releases')[0] if 'github.com/' in repo_url else repo_url
    
    # Process all pages until we reach the lookback date or run out of pages
    page_num = 1
    all_releases = []
    reached_lookback_date = False
    
    while not reached_lookback_date:
        current_url = f"{base_url}?page={page_num}"
        try:
            print(f"Fetching page {page_num}: {current_url}")
            response = requests.get(current_url, headers=headers)
            response.raise_for_status()
            html_content = response.text
            
            if raw_html and debug:
                # Save the full HTML page for debugging
                if not os.path.exists('debug_html'):
                    os.makedirs('debug_html')
                with open(f"debug_html/page_{page_num}_full.html", "w", encoding='utf-8') as f:
                    f.write(html_content)
                    if debug:
                        print(f"  Saved full HTML to debug_html/page_{page_num}_full.html")
            
            # Look for page empty indicator
            if "This repository hasn't published any releases" in html_content or "No releases published" in html_content:
                print("No releases found on this repository")
                break
            
            # Improved approach: Extract release sections by looking for complete release blocks
            release_blocks = []
            
            # First try to find release entry blocks with defined structure
            release_entries = re.findall(r'<div class="d-flex flex-column flex-md-row my-5[^>]*>.*?</section>', html_content, re.DOTALL)
            
            if release_entries:
                for entry in release_entries:
                    release_blocks.append(entry)
            else:
                # Fallback to Box-row sections (GitHub's standard layout)
                box_rows = re.findall(r'<div[^>]*class="[^"]*Box-row[^"]*"[^>]*>(.*?)</div>\s*(?:</div>|<div)', html_content, re.DOTALL)
                for block in box_rows:
                    release_blocks.append(block)
            
            # Process each release block
            for block in release_blocks:
                # Extract version and tag
                version_match = re.search(r'href="[^"]*?/releases/tag/([^"]+)"[^>]*>([^<]+)</a>', block)
                if not version_match:
                    continue
                    
                tag = version_match.group(1)
                version = version_match.group(2).strip()
                
                # More flexible version recognition pattern
                # Standard pattern for most repositories
                valid_version = re.search(r'v?\d+\.\d+\.\d+|beta|alpha|rc|arabica|mocha|beefy', version, re.IGNORECASE)
                
                # Special pattern for Avalanche repositories
                if is_avalanche and not valid_version:
                    valid_version = re.search(r'Etna|Durango|Cascade|Banff|Cortina|Fuji', version, re.IGNORECASE)
                
                # Special pattern for Optimism repositories (handle component prefixes)
                if is_optimism and not valid_version:
                    valid_version = re.search(r'op-\w+/v\d+\.\d+\.\d+|v\d+\.\d+\.\d+', version, re.IGNORECASE)
                
                # Special pattern for OP Succinct
                if is_succinct and not valid_version:
                    valid_version = re.search(r'op-succinct-v\d+\.\d+\.\d+|v\d+\.\d+\.\d+', version, re.IGNORECASE)
                
                if not valid_version:
                    continue
                
                # Extract the date directly from the same release block
                release_date = None
                date_str = None
                date_source = "unknown"
                
                # First try to find the relative time directly in the same block
                relative_time_match = re.search(r'<relative-time[^>]*datetime="([^"]+)"[^>]*>([^<]+)</relative-time>', block)
                if relative_time_match:
                    iso_date_str = relative_time_match.group(1)
                    display_date = relative_time_match.group(2).strip()
                    
                    try:
                        # GitHub uses ISO format with Z for UTC
                        if 'T' in iso_date_str:
                            iso_date_str = iso_date_str.replace('Z', '+00:00')
                            dt = datetime.fromisoformat(iso_date_str)
                            # Convert to local time for display
                            release_date = dt.replace(tzinfo=timezone.utc).astimezone(tz=None).replace(tzinfo=None)
                            date_source = f"ISO datetime: {iso_date_str} ({display_date})"
                    except (ValueError, TypeError):
                        pass
                
                # Also look for text date indicators like "3 hours ago" or "yesterday"
                if not release_date:
                    # Look for a time indicator in a clean version of the block
                    text_only = re.sub(r'<[^>]+>', ' ', block)
                    text_only = re.sub(r'\s+', ' ', text_only).strip()
                    
                    date_patterns = [
                        r'(\d+\s+(?:hours?|days?|weeks?|months?|years?)\s+ago)',  # 3 hours ago
                        r'(yesterday|last\s+(?:week|month|year))',                # yesterday
                        r'([A-Za-z]{3}\s+\d{1,2}(?:,\s*\d{4})?)',                # Feb 25
                        r'([A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?)'               # February 25
                    ]
                    
                    for pattern in date_patterns:
                        date_matches = re.findall(pattern, text_only)
                        for match in date_matches:
                            parsed_date = parse_relative_time(match, today) or parse_date_text(match, today)
                            if parsed_date and parsed_date <= today:
                                release_date = parsed_date
                                date_source = f"text: {match}"
                                break
                        
                        if release_date:
                            break
                
                # If still no date, look for a standalone date in the block
                if not release_date:
                    # This specifically targets the date display in the GitHub release layout
                    date_div_match = re.search(r'<div class="mb-2 f4[^>]*>([^<]+)</div>', block)
                    if date_div_match:
                        date_text = date_div_match.group(1).strip()
                        parsed_date = parse_date_text(date_text, today) or parse_relative_time(date_text, today)
                        if parsed_date:
                            release_date = parsed_date
                            date_source = f"date div: {date_text}"
                
                # Format the date if we found one
                if release_date:
                    date_str = release_date.strftime('%Y-%m-%d')
                    
                    # Check if this release is older than our lookback date
                    if release_date < lookback_date:
                        print(f"  Skipping {version} (released before {lookback_date.strftime('%Y-%m-%d')})")
                        reached_lookback_date = True
                        continue
                else:
                    # If we couldn't find a date, estimate based on position
                    # This is a fallback but should rarely be needed with the improved extraction
                    if page_num == 1:
                        release_date = today - timedelta(days=3)
                        date_source = "estimated recent (first page)"
                    else:
                        # Later pages - use position to estimate
                        position = release_blocks.index(block)
                        est_days = ((page_num - 1) * 10 + position) * 5  # Rough estimate
                        release_date = today - timedelta(days=est_days)
                        date_source = f"page position estimate: page {page_num}, pos {position}"
                    
                    date_str = release_date.strftime('%Y-%m-%d')
                
                # Determine if mandatory or optional
                release_type = 'Unknown'
                lower_context = block.lower()
                
                # Check for mandatory indicators
                if any(term in lower_context for term in [
                    'mandatory', 'required', 'security patch', 'critical', 'must upgrade', 
                    'vulnerability', 'security fix', 'critical fix', 'important fix',
                    'critical update', 'required update', 'security update'
                ]):
                    release_type = 'Mandatory'
                # Check for optional indicators
                elif any(term in lower_context for term in [
                    'optional', 'recommended', 'can upgrade', 'feature release', 
                    'enhancements', 'improvement', 'minor fix', 'quality of life',
                    'optional update', 'minor update', 'feature update'
                ]):
                    release_type = 'Optional'
                
                # Check if it's a pre-release
                is_pre_release = (
                    'pre-release' in lower_context or
                    '<span[^>]*>Pre-release</span>' in block or
                    re.search(r'class="[^"]*Label[^"]*">Pre', block) is not None or
                    'beta' in version.lower() or 
                    'alpha' in version.lower() or 
                    'rc' in version.lower() or
                    'preview' in version.lower() or
                    'arabica' in version.lower() or
                    'mocha' in version.lower() or
                    'beefy' in version.lower() or
                    '-bn' in version.lower()
                )
                
                # Extract a short description
                description = ""
                desc_match = re.search(r'<div[^>]*class="[^"]*markdown-body[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
                if desc_match:
                    raw_desc = desc_match.group(1)
                    # Remove HTML tags
                    clean_desc = re.sub(r'<[^>]+>', ' ', raw_desc)
                    # Clean up whitespace
                    clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                    description = clean_desc
                
                # If no description found, try to get it from the release page
                if not description:
                    try:
                        release_url = f"{base_url}/tag/{tag}"
                        release_resp = requests.get(release_url, headers=headers)
                        if release_resp.status_code == 200:
                            release_html = release_resp.text
                            desc_match = re.search(r'<div[^>]*class="[^"]*markdown-body[^"]*"[^>]*>(.*?)</div>', release_html, re.DOTALL)
                            if desc_match:
                                raw_desc = desc_match.group(1)
                                clean_desc = re.sub(r'<[^>]+>', ' ', raw_desc)
                                clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                                description = clean_desc
                    except Exception as e:
                        if debug:
                            print(f"  Error fetching release page: {e}")
                
                # If still no description, use a generic one
                if not description:
                    description = f"Release {version}"
                
                # Limit description length
                if len(description) > 200:
                    description = description[:197] + "..."
                
                print(f"  Found: {version} | {date_str} | {release_type} | {'Pre-release' if is_pre_release else 'Stable'} | Source: {date_source}")
                
                # Add to releases if not already there
                if not any(r['version'] == version for r in all_releases):
                    all_releases.append({
                        'version': version,
                        'releaseDate': date_str,
                        'type': release_type,
                        'isPreRelease': is_pre_release,
                        'description': description
                    })
            
            # Check if we need to stop pagination
            if reached_lookback_date:
                break
                
            if not release_blocks:
                if f"?page={page_num+1}" not in html_content and "Next" not in html_content:
                    print("No more pages available")
                    break
            
            # Move to next page
            page_num += 1
            
            # Add delay if rate limiting is enabled
            if rate_limit and page_num > 1:
                import time
                time.sleep(1)  # 1 second delay
            
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Sort by date (newest first)
    all_releases.sort(key=lambda x: x['releaseDate'], reverse=True)
    
    result = {
        'repository': repo_name,
        'fullUrl': base_url,
        'timeframe': {
            'from': lookback_date.strftime('%Y-%m-%d'),
            'to': today.strftime('%Y-%m-%d')
        },
        'totalReleases': len(all_releases),
        'releases': all_releases
    }
    
    # Generate default output filename if none provided
    if not output_file:
        repo_short_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
        output_file = f"{repo_short_name}_releases.json"
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Data saved to {output_file}")
    
    # Print summary
    print("\n=== RELEASE SUMMARY ===")
    print(f"Repository: {repo_name}")
    print(f"Timeframe: {lookback_date.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")
    print(f"Total releases: {len(all_releases)}")
    
    mandatory_count = sum(1 for r in all_releases if r['type'] == 'Mandatory')
    optional_count = sum(1 for r in all_releases if r['type'] == 'Optional')
    unknown_count = sum(1 for r in all_releases if r['type'] == 'Unknown')
    
    print(f"Mandatory releases: {mandatory_count}")
    print(f"Optional releases: {optional_count}")
    print(f"Unknown type: {unknown_count}")
    
    if len(all_releases) > 1:
        dates = [datetime.strptime(r['releaseDate'], '%Y-%m-%d') for r in all_releases]
        time_diffs = [(dates[i] - dates[i+1]).days for i in range(len(dates)-1)]
        if time_diffs:
            avg_days = sum(time_diffs) / len(time_diffs)
            print(f"Average days between releases: {avg_days:.1f}")
    
    print("\n=== RELEASE DETAILS ===")
    for release in all_releases:
        print(f"{release['version']} | {release['releaseDate']} | {release['type']} | {'Pre-release' if release['isPreRelease'] else 'Stable'}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract GitHub release data with improved date recognition')
    parser.add_argument('repo_url', help='GitHub repository URL')
    parser.add_argument('--months', type=int, default=6, help='Number of months to look back')
    parser.add_argument('--output', help='Output JSON file (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--raw-html', action='store_true', help='Save raw HTML for debugging')
    parser.add_argument('--rate-limit', action='store_true', help='Add delay to avoid rate limiting')
    args = parser.parse_args()
    
    extract_github_releases(
        args.repo_url, 
        args.months, 
        args.output, 
        args.debug, 
        args.raw_html,
        args.rate_limit
    )

if __name__ == "__main__":
    main()