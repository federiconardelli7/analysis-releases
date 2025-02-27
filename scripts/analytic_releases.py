#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import os
import numpy as np
from datetime import datetime, timedelta

# Add this class to handle NumPy data types for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Period):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

def analyze_github_releases(json_file, output_dir="release_analysis"):
    """
    Analyze GitHub release data from JSON file and create visualizations
    
    Args:
        json_file (str): Path to the JSON file with release data
        output_dir (str): Directory to save output files and images
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    repo_name = data['repository']
    print(f"Analyzing releases for {repo_name}")
    
    # Convert releases to DataFrame
    df = pd.DataFrame(data['releases'])
    
    # Convert releaseDate to datetime for better analysis
    df['releaseDate'] = pd.to_datetime(df['releaseDate'])
    
    # Extract network information from version (arabica, mocha, etc.)
    def extract_network(version):
        if 'arabica' in version.lower():
            return 'arabica'
        elif 'mocha' in version.lower():
            return 'mocha'
        elif 'beefy' in version.lower():
            return 'beefy'
        elif 'etna' in version.lower():
            return 'etna'
        elif 'durango' in version.lower():
            return 'durango'
        elif 'cascade' in version.lower():
            return 'cascade'
        elif 'banff' in version.lower():
            return 'banff'
        elif 'op-node' in version.lower():
            return 'op-node'
        elif 'op-batcher' in version.lower():
            return 'op-batcher'
        elif 'op-challenger' in version.lower():
            return 'op-challenger'
        elif 'op-contracts' in version.lower():
            return 'op-contracts'
        elif 'succinct' in version.lower():
            return 'succinct'
        else:
            return 'mainnet'
    
    df['network'] = df['version'].apply(extract_network)
    
    # Extract semantic versioning components
    def extract_semver(version):
        # First try regular semantic versioning
        match = re.search(r'v?(\d+)\.(\d+)\.(\d+)', version)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        
        # Then try Avalanche-style versioning (e.g., Etna.2)
        match = re.search(r'([A-Za-z]+)\.(\d+)', version)
        if match:
            # Use network name as major version and the number as minor version
            # Convert network name to a number using the first letter's ASCII value
            network_name = match.group(1)
            network_value = ord(network_name[0].lower()) - ord('a') + 1  # 'a' becomes 1, 'b' becomes 2, etc.
            return network_value, int(match.group(2)), 0
        
        return None, None, None
    
    df['major'], df['minor'], df['patch'] = zip(*df['version'].apply(extract_semver))
    
    # Create a column for the month-year for grouping
    df['month_year'] = df['releaseDate'].dt.to_period('M')
    
    # Calculate time between releases
    df = df.sort_values('releaseDate', ascending=True).reset_index(drop=True)
    df['days_since_last_release'] = (df['releaseDate'] - df['releaseDate'].shift(1)).dt.days
    
    # ===== Generate Analytics =====
    
    # 1. Releases per month visualization
    plt.figure(figsize=(12, 6))
    monthly_releases = df.groupby('month_year').size()
    monthly_releases.plot(kind='bar', color='skyblue')
    plt.title(f'Number of Releases per Month - {repo_name}')
    plt.xlabel('Month')
    plt.ylabel('Number of Releases')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_releases.png")
    
    # 2. Release types distribution
    plt.figure(figsize=(10, 6))
    type_counts = df['type'].value_counts()
    type_counts.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    plt.title(f'Distribution of Release Types - {repo_name}')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/release_types.png")
    
    # 3. Pre-release vs Stable releases
    plt.figure(figsize=(8, 6))
    prerelease_counts = df['isPreRelease'].value_counts()
    prerelease_counts.plot(kind='bar', color=['#ff9999', '#66b3ff'])
    plt.title(f'Pre-release vs Stable Releases - {repo_name}')
    plt.xlabel('Is Pre-release')
    plt.ylabel('Number of Releases')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prerelease_vs_stable.png")
    
    # 4. Network distribution
    plt.figure(figsize=(10, 6))
    network_counts = df['network'].value_counts()
    network_counts.plot(kind='bar', color=sns.color_palette("viridis", len(network_counts)))
    plt.title(f'Distribution of Networks - {repo_name}')
    plt.xlabel('Network')
    plt.ylabel('Number of Releases')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/network_distribution.png")
    
    # 5. Release frequency over time (cumulative)
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('releaseDate')
    plt.plot(df_sorted['releaseDate'], range(1, len(df) + 1), marker='o', linestyle='-')
    plt.title(f'Cumulative Releases Over Time - {repo_name}')
    plt.xlabel('Date')
    plt.ylabel('Total Number of Releases')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_releases.png")
    
    # 6. Days between releases histogram
    plt.figure(figsize=(10, 6))
    df['days_since_last_release'].dropna().plot(kind='hist', bins=15, color='skyblue', edgecolor='black')
    plt.title(f'Days Between Releases - {repo_name}')
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/days_between_releases.png")
    
    # 7. Major version distribution
    plt.figure(figsize=(10, 6))
    df_valid = df[df['major'].notnull()]
    version_counts = df_valid.groupby(['major', 'minor']).size().unstack(fill_value=0)
    version_counts.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title(f'Distribution by Major & Minor Versions - {repo_name}')
    plt.xlabel('Major Version')
    plt.ylabel('Number of Releases')
    plt.xticks(rotation=0)
    plt.legend(title='Minor Version')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/version_distribution.png")
    
    # 8. Timeline of releases by network
    plt.figure(figsize=(14, 8))
    networks = df['network'].unique()
    colors = sns.color_palette("husl", len(networks))
    
    for i, network in enumerate(networks):
        network_df = df[df['network'] == network]
        plt.scatter(network_df['releaseDate'], [i] * len(network_df), 
                   label=network, s=100, color=colors[i], alpha=0.7)
        
        # Add version labels
        for j, (date, version) in enumerate(zip(network_df['releaseDate'], network_df['version'])):
            plt.text(date, i + 0.1, version, rotation=45, fontsize=8, ha='center', va='bottom')
    
    plt.yticks(range(len(networks)), networks)
    plt.title(f'Timeline of Releases by Network - {repo_name}')
    plt.xlabel('Date')
    plt.ylabel('Network')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/release_timeline.png")
    
    # ===== Generate Statistics Report =====
    
    stats = {
        'repository': repo_name,
        'total_releases': len(df),
        'date_range': {
            'first_release': df['releaseDate'].min().strftime('%Y-%m-%d'),
            'last_release': df['releaseDate'].max().strftime('%Y-%m-%d'),
            'days_span': (df['releaseDate'].max() - df['releaseDate'].min()).days
        },
        'release_types': {
            'mandatory': int(df['type'].eq('Mandatory').sum()),
            'optional': int(df['type'].eq('Optional').sum()),
            'unknown': int(df['type'].eq('Unknown').sum())
        },
        'release_stability': {
            'stable': int(df['isPreRelease'].eq(False).sum()),
            'pre_release': int(df['isPreRelease'].eq(True).sum())
        },
        'networks': {
            network: int(count) for network, count in df['network'].value_counts().items()
        },
        'version_stats': {
            'major_versions': list(df[df['major'].notnull()]['major'].unique()),
            'latest_version': df.loc[0, 'version']
        },
        'time_metrics': {
            'avg_days_between_releases': round(df['days_since_last_release'].mean(), 1),
            'median_days_between_releases': round(df['days_since_last_release'].median(), 1),
            'min_days_between_releases': int(df['days_since_last_release'].min()),
            'max_days_between_releases': int(df['days_since_last_release'].max()),
            'releases_per_month': round(len(df) / (df['month_year'].nunique()), 1)
        }
    }
    
    # Save statistics to JSON - use the custom encoder here
    with open(f"{output_dir}/release_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    
    # Generate a text report
    report = f"""
    =========================================================
    GitHub Release Analysis Report for {repo_name}
    =========================================================
    
    OVERVIEW:
    ---------------------------------------------------------
    Total Releases: {stats['total_releases']}
    Date Range: {stats['date_range']['first_release']} to {stats['date_range']['last_release']} ({stats['date_range']['days_span']} days)
    Latest Version: {stats['version_stats']['latest_version']}
    
    RELEASE TYPES:
    ---------------------------------------------------------
    Mandatory: {stats['release_types']['mandatory']} ({stats['release_types']['mandatory']/stats['total_releases']*100:.1f}%)
    Optional: {stats['release_types']['optional']} ({stats['release_types']['optional']/stats['total_releases']*100:.1f}%)
    Unknown: {stats['release_types']['unknown']} ({stats['release_types']['unknown']/stats['total_releases']*100:.1f}%)
    
    STABILITY:
    ---------------------------------------------------------
    Stable Releases: {stats['release_stability']['stable']} ({stats['release_stability']['stable']/stats['total_releases']*100:.1f}%)
    Pre-releases: {stats['release_stability']['pre_release']} ({stats['release_stability']['pre_release']/stats['total_releases']*100:.1f}%)
    
    NETWORKS:
    ---------------------------------------------------------
    {chr(10).join([f"{network}: {count} releases ({count/stats['total_releases']*100:.1f}%)" for network, count in stats['networks'].items()])}
    
    TIME METRICS:
    ---------------------------------------------------------
    Average days between releases: {stats['time_metrics']['avg_days_between_releases']}
    Median days between releases: {stats['time_metrics']['median_days_between_releases']}
    Shortest release cycle: {stats['time_metrics']['min_days_between_releases']} days
    Longest release cycle: {stats['time_metrics']['max_days_between_releases']} days
    Releases per month (average): {stats['time_metrics']['releases_per_month']}
    
    RECENT RELEASES (LAST 5):
    ---------------------------------------------------------
    {chr(10).join([f"{row['releaseDate'].strftime('%Y-%m-%d')} - {row['version']} - {row['network']} network" for _, row in df.head(5).iterrows()])}
    
    =========================================================
    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Analysis images saved to '{output_dir}' directory
    =========================================================
    """
    
    # Save the report to a text file
    with open(f"{output_dir}/release_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"Analysis complete! Report and visualizations saved to '{output_dir}' directory.")
    
    return stats, df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze GitHub release data from JSON file')
    parser.add_argument('json_file', help='Path to the JSON file with release data')
    parser.add_argument('--output-dir', default='release_analysis', help='Directory to save output files and images')
    args = parser.parse_args()
    
    analyze_github_releases(args.json_file, args.output_dir)

if __name__ == "__main__":
    main() 