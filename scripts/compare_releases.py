#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime, timedelta
import re
import numpy as np

def load_stats(project_name, stats_file):
    """Load statistics from a JSON file"""
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            stats['project'] = project_name  # Add project name to stats
            return stats
    except FileNotFoundError:
        print(f"ERROR: Statistics file not found: {stats_file}")
        print(f"Make sure to run celestia_release_analytics.py for {project_name} first!")
        return None

def create_comparison_report(stats_list, output_dir="comparison_analysis"):
    """Create comparison visualizations and report"""
    # Filter out None values (failed loads)
    stats_list = [stats for stats in stats_list if stats is not None]
    
    if not stats_list:
        print("No valid statistics files were loaded. Cannot generate comparison.")
        return
    
    # Extract months value from environment or use a default
    try:
        months_lookback = int(os.environ.get('MONTHS', '6'))
    except (ValueError, TypeError):
        months_lookback = 6  # Default if not specified or invalid
    
    timeframe_text = f"(Last {months_lookback} Months)"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(stats_list)
    
    # Create a custom color palette that ensures Avalanche is always red
    def get_project_colors(projects):
        # Use a standard color palette as a base
        standard_colors = sns.color_palette("husl", len(projects))
        
        # Create a dictionary to map projects to colors
        color_map = {}
        
        # Assign colors to each project
        for i, project in enumerate(projects):
            if project == "Avalanche":
                color_map[project] = (0.8, 0.2, 0.2)  # Red for Avalanche
            else:
                color_map[project] = standard_colors[i]
        
        return color_map

    # Get the list of projects from stats_list
    project_names = [stats['project'] for stats in stats_list]
    project_colors = get_project_colors(project_names)
    
    # Create project comparison visualizations
    
    # 1. Total Releases Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='project', y='total_releases', data=df, palette=project_colors)
    plt.title(f'Total Releases by Project {timeframe_text}')
    plt.ylabel('Number of Releases')
    plt.xlabel('Project')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_releases_comparison.png")
    
    # 2. Release Frequency (Avg Days Between Releases)
    plt.figure(figsize=(10, 6))
    freq_data = [(p, stats['time_metrics']['avg_days_between_releases']) 
                 for p, stats in zip(df['project'], df.to_dict('records'))]
    freq_df = pd.DataFrame(freq_data, columns=['project', 'avg_days'])
    sns.barplot(x='project', y='avg_days', data=freq_df, palette=project_colors)
    plt.title(f'Average Days Between Releases {timeframe_text}')
    plt.ylabel('Days')
    plt.xlabel('Project')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/release_frequency_comparison.png")
    
    # 3. Mandatory vs Optional vs Unknown
    plt.figure(figsize=(12, 6))
    mandatory_data = []
    optional_data = []
    unknown_data = []
    
    for p, stats in zip(df['project'], df.to_dict('records')):
        total = stats['total_releases']
        mandatory_pct = stats['release_types']['mandatory'] / total * 100
        optional_pct = stats['release_types']['optional'] / total * 100
        unknown_pct = stats['release_types']['unknown'] / total * 100
        
        mandatory_data.append((p, mandatory_pct))
        optional_data.append((p, optional_pct))
        unknown_data.append((p, unknown_pct))
    
    types_df = pd.DataFrame({
        'project': [p for p, _ in mandatory_data],
        'Mandatory': [pct for _, pct in mandatory_data],
        'Optional': [pct for _, pct in optional_data],
        'Unknown': [pct for _, pct in unknown_data]
    })
    
    types_df.plot(x='project', kind='bar', stacked=True, 
                 color=['#ff9999', '#66b3ff', '#c2c2f0'],
                 figsize=(12, 6))
    plt.title(f'Release Types Distribution by Project {timeframe_text}')
    plt.ylabel('Percentage')
    plt.xlabel('Project')
    plt.legend(title='Release Type')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/release_types_comparison.png")
    
    # 4. Stable vs Pre-release
    plt.figure(figsize=(12, 6))
    stability_data = []
    
    for p, stats in zip(df['project'], df.to_dict('records')):
        total = stats['total_releases']
        stable_pct = stats['release_stability']['stable'] / total * 100
        prerelease_pct = stats['release_stability']['pre_release'] / total * 100
        
        stability_data.append((p, stable_pct, prerelease_pct))
    
    stability_df = pd.DataFrame({
        'project': [p for p, _, _ in stability_data],
        'Stable': [s for _, s, _ in stability_data],
        'Pre-release': [pr for _, _, pr in stability_data]
    })
    
    stability_df.plot(x='project', kind='bar', stacked=True, 
                     color=['#66b3ff', '#ff9999'],
                     figsize=(12, 6))
    plt.title(f'Release Stability by Project {timeframe_text}')
    plt.ylabel('Percentage')
    plt.xlabel('Project')
    plt.legend(title='Stability')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_comparison.png")
    
    # 5. Releases per Month
    plt.figure(figsize=(16, 10))

    # Collect release data for all projects
    all_project_releases = {}
    earliest_date = datetime.now()
    latest_date = datetime(2000, 1, 1)
    
    # First pass to get date ranges
    for project_stats in stats_list:
        project = project_stats['project']
        first_date = datetime.strptime(project_stats['date_range']['first_release'], '%Y-%m-%d')
        last_date = datetime.strptime(project_stats['date_range']['last_release'], '%Y-%m-%d')
        
        if first_date < earliest_date:
            earliest_date = first_date
        if last_date > latest_date:
            latest_date = last_date
            
    # Add some padding to the date range
    earliest_date = earliest_date - timedelta(days=14)
    latest_date = latest_date + timedelta(days=14)
    
    # Load the detailed release data for each project
    for project_stats in stats_list:
        project = project_stats['project']
        try:
            # Find the source JSON file
            possible_json_files = [
                f"jsons/{project.lower().replace(' ', '_')}_releases.json",
                f"jsons/{project.lower().replace(' ', '_')}-node_releases.json",
                f"jsons/op_succinct_releases.json" if project == "OP Succinct" else None,
                # Add fallbacks for backwards compatibility
                f"{project.lower().replace(' ', '_')}_releases.json", 
                f"{project.lower().replace(' ', '_')}-node_releases.json"
            ]
            
            json_file = None
            for file in possible_json_files:
                if os.path.exists(file):
                    json_file = file
                    break
                    
            if not json_file:
                print(f"Warning: Could not find release data JSON for {project}. Skipping timeline visualization.")
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            releases = data['releases']
            dates = [datetime.strptime(r['releaseDate'], '%Y-%m-%d') for r in releases]
            dates.sort()  # Ensure sorted order
            
            all_project_releases[project] = dates
            
        except Exception as e:
            print(f"Error loading release data for {project}: {e}")
    
    # Ensure all projects span the entire date range by adding placeholder dates if needed
    for project in all_project_releases:
        dates = all_project_releases[project]
        if dates and max(dates) < latest_date - timedelta(days=30):
            print(f"Note: {project} has no recent releases (last: {max(dates).strftime('%Y-%m-%d')})")
            # Don't add artificial dates, but ensure visualizations handle this correctly
    
    # Create a cumulative releases visualization
    plt.figure(figsize=(14, 8))
    
    for project, dates in all_project_releases.items():
        # Create cumulative counts
        cumulative_dates = sorted(dates)
        
        # Get counts by date
        date_range = pd.date_range(earliest_date, latest_date, freq='D')
        counts = pd.Series(0, index=date_range)
        
        for d in cumulative_dates:
            if d in counts.index:
                counts[d] += 1
        
        # Cumulative sum
        cumulative = counts.cumsum()
        
        # Plot with consistent colors (Avalanche in red)
        plt.plot(cumulative.index, cumulative.values, marker='', linestyle='-', linewidth=2.5, 
                 color=project_colors[project], label=project)
    
    plt.title('Cumulative Releases by Project ' + timeframe_text)
    plt.xlabel('Date')
    plt.ylabel('Total Number of Releases')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_releases_comparison.png")
    
    # Create a unified release timeline visualization
    plt.figure(figsize=(16, 10))
    
    # Determine y-positions for each project for better visualization
    y_positions = {project: i for i, project in enumerate(all_project_releases.keys())}
    
    for project, dates in all_project_releases.items():
        y_pos = y_positions[project]
        for date in dates:
            plt.scatter(date, y_pos, s=100, alpha=0.7, 
                       color=project_colors[project])
        
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    plt.title('Release Timeline Comparison ' + timeframe_text)
    plt.xlabel('Date')
    plt.ylabel('Project')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add month markers on x-axis
    months = pd.date_range(earliest_date, latest_date, freq='MS')
    plt.xticks(months, [d.strftime('%b %Y') for d in months], rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/release_timeline_comparison.png")
    
    # Release Frequency Over Time (releases per month)
    plt.figure(figsize=(14, 8))
    
    # Set a fixed date range for all projects
    date_range = pd.date_range(start=earliest_date, end=latest_date, freq='MS')  # Month start dates with no padding
    month_labels = [d.strftime('%Y-%m') for d in date_range]

    # Create a DataFrame to hold all projects' monthly release counts
    monthly_data = pd.DataFrame(index=month_labels)

    print("\nDetailed monthly release counts by project:")
    for project, dates in all_project_releases.items():
        # Count releases by month using string format for consistency
        month_counts = {}
        for date in dates:
            month_key = date.strftime('%Y-%m')
            month_counts[month_key] = month_counts.get(month_key, 0) + 1
        
        # Create a complete series with all months
        complete_series = []
        for month in month_labels:
            complete_series.append(month_counts.get(month, 0))
        
        # Add to the dataframe    
        monthly_data[project] = complete_series
        
        # Print detailed debug for this project
        print(f"\n{project} release counts by month:")
        for month, count in zip(month_labels, complete_series):
            if count > 0:
                print(f"  {month}: {count}")

    # Calculate 3-month rolling average for each project
    plot_data = monthly_data.copy()
    for project in plot_data.columns:
        # Calculate rolling average with min_periods=1 to handle edges properly
        plot_data[project] = plot_data[project].rolling(window=3, center=True, min_periods=1).mean()

    # Convert to proper datetime for x-axis
    plot_dates = [datetime.strptime(m, '%Y-%m') for m in month_labels]

    # Plot each project
    for project in plot_data.columns:
        # First plot raw data points as faint background
        for i, count in enumerate(monthly_data[project]):
            if count > 0:
                plt.scatter(plot_dates[i], count, s=30, alpha=0.3, color=project_colors[project])
        
        # Then plot the rolling average line
        plt.plot(plot_dates, plot_data[project], 
                marker='o', markersize=5, linewidth=2.5, alpha=0.8,
                color=project_colors[project], label=project)

    # Set the x-axis ticks to show each month
    plt.xticks(plot_dates, [d.strftime('%Y-%m') for d in plot_dates], rotation=45)
    plt.xlim(plot_dates[0], plot_dates[-1])  # Ensure all dates are shown

    plt.title('Monthly Release Frequency by Project ' + timeframe_text)
    plt.xlabel('Date')
    plt.ylabel('Releases per Month')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/release_frequency_over_time.png")
    
    # Stability Trend Over Time - Fix
    plt.figure(figsize=(14, 8))

    for project, dates in all_project_releases.items():
        # Try to find the corresponding JSON file directly, rather than using complex path substitutions
        json_file = None
        for possible_file in [
            f"jsons/{project.lower().replace(' ', '_')}_releases.json", 
            f"jsons/{project.lower().replace(' ', '_')}-node_releases.json",
            f"jsons/op_succinct_releases.json" if project == "OP Succinct" else None,
            # Add fallbacks for backwards compatibility
            f"{project.lower().replace(' ', '_')}_releases.json", 
            f"{project.lower().replace(' ', '_')}-node_releases.json"
        ]:
            if possible_file and os.path.exists(possible_file):
                json_file = possible_file
                break
        
        if not json_file:
            print(f"Warning: Could not find release data JSON for {project} stability trend.")
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            releases = data['releases']
            
            # Create a DataFrame with dates and pre-release status
            release_dates = [datetime.strptime(r['releaseDate'], '%Y-%m-%d') for r in releases]
            is_prerelease = [r['isPreRelease'] for r in releases]
            
            # Sort by date
            release_data = sorted(zip(release_dates, is_prerelease), key=lambda x: x[0])
            
            # Apply a rolling window to smooth the data
            window_size = 5  # Adjust as needed
            prerelease_values = [int(is_pre) for _, is_pre in release_data]
            
            # Use manual rolling window calculation
            smoothed_values = []
            for i in range(len(prerelease_values)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(prerelease_values), i + window_size // 2 + 1)
                window_values = prerelease_values[start_idx:end_idx]
                smoothed_values.append(sum(window_values) / len(window_values) * 100)
            
            plt.plot([d for d, _ in release_data], smoothed_values, 
                    marker='o', markersize=4, linewidth=2, alpha=0.7,
                    color=project_colors[project],
                    label=f"{project}")
        
        except Exception as e:
            print(f"Error creating stability trend for {project}: {e}")

    plt.title('Pre-release Percentage by Project Over Time ' + timeframe_text)
    plt.xlabel('Date')
    plt.ylabel('Percentage of Pre-releases')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stability_trend.png")
    
    # Version Evolution Speed
    plt.figure(figsize=(12, 6))
    version_change_data = []

    for project_stats in stats_list:
        project = project_stats['project']
        try:
            json_file = None
            # Look for the JSON files in the jsons/ directory
            for possible_file in [
                f"jsons/{project.lower().replace(' ', '_')}_releases.json", 
                f"jsons/{project.lower().replace(' ', '_')}-node_releases.json",
                f"jsons/op_succinct_releases.json" if project == "OP Succinct" else None,
                # Add fallbacks for backwards compatibility
                f"{project.lower().replace(' ', '_')}_releases.json", 
                f"{project.lower().replace(' ', '_')}-node_releases.json"
            ]:
                if possible_file and os.path.exists(possible_file):
                    json_file = possible_file
                    break
                    
            if not json_file:
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            releases = data['releases']
            
            # Calculate average days between major/minor version bumps
            version_changes = []
            last_major = None
            last_minor = None
            last_date = None
            
            for release in sorted(releases, key=lambda x: x['releaseDate']):
                date = datetime.strptime(release['releaseDate'], '%Y-%m-%d')
                # Extract version numbers using regex
                match = re.search(r'v?(\d+)\.(\d+)', release['version'])
                
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    
                    if last_major is not None and last_date is not None:
                        days_since = (date - last_date).days
                        
                        if major > last_major:
                            version_changes.append(('major', days_since))
                        elif minor > last_minor:
                            version_changes.append(('minor', days_changes))
                            
                    last_major, last_minor, last_date = major, minor, date
            
            # Calculate averages
            if version_changes:
                major_changes = [days for type, days in version_changes if type == 'major']
                minor_changes = [days for type, days in version_changes if type == 'minor']
                
                avg_major = sum(major_changes) / len(major_changes) if major_changes else 0
                avg_minor = sum(minor_changes) / len(minor_changes) if minor_changes else 0
                
                version_change_data.append({
                    'project': project,
                    'avg_days_between_major': avg_major,
                    'avg_days_between_minor': avg_minor
                })
        except Exception as e:
            print(f"Error analyzing version evolution for {project}: {e}")

    if version_change_data:
        version_df = pd.DataFrame(version_change_data)
        version_df = version_df.set_index('project')
        
        version_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')
        plt.title('Average Days Between Version Changes ' + timeframe_text)
        plt.xlabel('Project')
        plt.ylabel('Days')
        plt.legend(['Major Version Changes', 'Minor Version Changes'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/version_evolution_speed.png")
    
    # Release Consistency Analysis (Variation in Release Cadence)
    plt.figure(figsize=(12, 6))
    cadence_data = []

    for project, dates in all_project_releases.items():
        if len(dates) > 2:  # Need at least 3 releases to calculate intervals
            sorted_dates = sorted(dates)
            intervals = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
            
            # Calculate standard deviation of intervals
            std_dev = np.std(intervals)
            cv = std_dev / np.mean(intervals) * 100 if np.mean(intervals) > 0 else 0  # Coefficient of variation
            
            cadence_data.append({
                'project': project, 
                'std_dev': std_dev,
                'cv': cv  # Lower is more consistent
            })

    if cadence_data:
        cadence_df = pd.DataFrame(cadence_data)
        
        # Plot coefficient of variation (lower means more consistent)
        sns.barplot(x='project', y='cv', data=cadence_df, palette=project_colors)
        plt.title(f'Release Schedule Consistency by Project {timeframe_text}')
        plt.xlabel('Project')
        plt.ylabel('Coefficient of Variation (%) - Lower is More Consistent')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/release_consistency.png")

    # Replace the monthly release frequency visualization with a clearer heatmap version
    plt.figure(figsize=(16, 10))

    # Set fixed date range for all projects (pad by one month on each side)
    date_range = pd.date_range(
        start=earliest_date,
        end=latest_date, 
        freq='MS'
    )  # Month start dates with no padding
    month_labels = [d.strftime('%Y-%m') for d in date_range]

    # Create DataFrame for heatmap
    heatmap_data = []

    # Prepare data for all projects
    for project, dates in all_project_releases.items():
        # Count releases by month
        month_counts = {}
        for date in dates:
            month_key = date.strftime('%Y-%m')
            month_counts[month_key] = month_counts.get(month_key, 0) + 1
        
        # Fill in all months in the range
        for month in month_labels:
            count = month_counts.get(month, 0)
            heatmap_data.append({
                'Project': project,
                'Month': month,
                'Releases': count
            })

    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)

    # Pivot the data for the heatmap
    pivot_data = heatmap_df.pivot(index='Project', columns='Month', values='Releases')

    # Ensure proper ordering of projects (with Avalanche in a prominent position)
    project_order = [p for p in all_project_releases.keys()]
    pivot_data = pivot_data.reindex(project_order)

    # Create heatmap
    plt.figure(figsize=(18, 8))
    ax = sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', 
                    linewidths=0.5, cbar_kws={'label': 'Number of Releases'})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.title('Monthly Release Count by Project ' + timeframe_text, fontsize=14, pad=20)

    # Add grid lines to improve readability
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_releases_heatmap.png")

    # Fix the average monthly releases calculation to account for project timeframes
    plt.figure(figsize=(12, 7))

    # Calculate true monthly averages based on each project's active timeframe
    true_monthly_averages = []

    for project, dates in all_project_releases.items():
        if dates:
            # Sort dates and get first and last
            sorted_dates = sorted(dates)
            first_release = sorted_dates[0]
            last_release = sorted_dates[-1]
            
            # Count months between first and last release (add 1 to include both start and end months)
            months_active = (last_release.year - first_release.year) * 12 + (last_release.month - first_release.month) + 1
            
            # Ensure we don't divide by zero
            if months_active > 0:
                monthly_average = len(dates) / months_active
            else:
                monthly_average = len(dates)  # If all releases are in same month
            
            true_monthly_averages.append({
                'project': project,
                'avg_releases': monthly_average,
                'total_releases': len(dates),
                'months_active': months_active,
                'first_release': first_release.strftime('%Y-%m'),
                'last_release': last_release.strftime('%Y-%m')
            })

    if true_monthly_averages:
        # Convert to DataFrame and sort
        avg_df = pd.DataFrame(true_monthly_averages)
        avg_df = avg_df.sort_values('avg_releases', ascending=False)
        
        # Manually create bar chart to have better control of positions
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(range(len(avg_df)), avg_df['avg_releases'], alpha=0.8)
        
        # Set x-tick labels with proper rotation
        ax.set_xticks(range(len(avg_df)))
        ax.set_xticklabels(avg_df['project'], rotation=45, ha='right')
        
        # Add colors to bars using project_colors
        for i, bar in enumerate(bars):
            project = avg_df.iloc[i]['project']
            bar.set_color(project_colors[project])
        
        # For the integrated view
        for i, (idx, row) in enumerate(avg_df.iterrows()):
            project = idx
            active_period = f"{row['first_release']} to {row['last_release']}"
            months_info = f"({row['months_active']} months)"
            
            # Only add text annotation if the bar is tall enough
            if row['avg_releases'] > 0.7:  # Only add text if bar is tall enough
                ax.annotate(
                    f"{active_period}\n{months_info}",
                    xy=(i, row['avg_releases'] / 2),  # Position in the middle of the bar
                    ha='center', va='center',  # Center alignment
                    fontsize=7,
                    color='white',  # White text for contrast
                    weight='bold',  # Make it bold for better readability
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.3)
                )
            else:
                # For short bars, put the text above the bar
                ax.annotate(
                    f"{active_period}\n{months_info}",
                    xy=(i, row['avg_releases']),
                    xytext=(0, 5),  # 5 points above
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
                )
        
        plt.title('Average Monthly Release Count During Active Period ' + timeframe_text)
        plt.ylabel('Average Releases per Month')
        plt.xlabel('Project')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/avg_monthly_releases.png")
        
        # Print detailed average information for reference
        print("\nDetailed monthly average release statistics:")
        for row in true_monthly_averages:
            print(f"{row['project']}: {row['avg_releases']:.2f} releases/month over {row['months_active']} months " +
                  f"({row['first_release']} to {row['last_release']}, {row['total_releases']} total releases)")

    # Create a quarterly view for better trend visibility (less noisy)
    quarterly_data = []
    for project, dates in all_project_releases.items():
        # Group dates by quarter
        quarters = {}
        for date in dates:
            quarter = f"{date.year}-Q{(date.month-1)//3 + 1}"
            quarters[quarter] = quarters.get(quarter, 0) + 1
        
        # Create quarterly data points
        for quarter, count in quarters.items():
            quarterly_data.append({
                'Project': project,
                'Quarter': quarter,
                'Releases': count
            })

    # Convert to DataFrame and plot
    q_df = pd.DataFrame(quarterly_data)

    # Only proceed if we have quarterly data
    if not q_df.empty:
        # Get unique quarters in chronological order
        all_quarters = sorted(q_df['Quarter'].unique())
        
        # Create figure for plotting
        plt.figure(figsize=(14, 8))
        
        # Plot a line for each project
        for project in all_project_releases:
            project_data = q_df[q_df['Project'] == project]
            
            # Create a complete series with all quarters
            quarters_series = []
            quarter_values = []
            for q in all_quarters:
                quarters_series.append(q)
                match = project_data[project_data['Quarter'] == q]
                quarter_values.append(match['Releases'].values[0] if not match.empty else 0)
            
            # Plot the line with the project's color
            plt.plot(quarters_series, quarter_values, marker='o', linewidth=2.5, 
                     markersize=8, label=project, color=project_colors[project])
        
        plt.title('Quarterly Release Trends by Project ' + timeframe_text)
        plt.xlabel('Quarter')
        plt.ylabel('Number of Releases')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quarterly_release_trend.png")
    
    # Generate comparison report text
    report = f"""
    =========================================================
    Layer 2 Projects Release Comparison Report
    =========================================================
    
    OVERVIEW:
    ---------------------------------------------------------
    Projects analyzed: {', '.join(df['project'])}
    Analysis date: {datetime.now().strftime('%Y-%m-%d')}
    
    RELEASE VOLUME:
    ---------------------------------------------------------
    {chr(10).join([f"{p}: {stats['total_releases']} total releases" for p, stats in zip(df['project'], df.to_dict('records'))])}
    
    RELEASE FREQUENCY:
    ---------------------------------------------------------
    {chr(10).join([f"{p}: {stats['time_metrics']['avg_days_between_releases']} days between releases (average)" for p, stats in zip(df['project'], df.to_dict('records'))])}
    {chr(10).join([f"{p}: {stats['time_metrics']['median_days_between_releases']} days between releases (median)" for p, stats in zip(df['project'], df.to_dict('records'))])}
    
    RELEASE TYPES:
    ---------------------------------------------------------
    {chr(10).join([f"{p}: {stats['release_types']['mandatory']} mandatory, {stats['release_types']['optional']} optional, {stats['release_types']['unknown']} unknown" for p, stats in zip(df['project'], df.to_dict('records'))])}
    
    STABILITY:
    ---------------------------------------------------------
    {chr(10).join([f"{p}: {stats['release_stability']['stable']} stable releases, {stats['release_stability']['pre_release']} pre-releases" for p, stats in zip(df['project'], df.to_dict('records'))])}
    
    PROJECT ACTIVITY:
    ---------------------------------------------------------
    {chr(10).join([f"{p}: Active for {stats['date_range']['days_span']} days from {stats['date_range']['first_release']} to {stats['date_range']['last_release']}" for p, stats in zip(df['project'], df.to_dict('records'))])}
    
    PROJECT MATURITY INDICATORS:
    ---------------------------------------------------------
    {chr(10).join([f"{p}: {len(stats['version_stats']['major_versions'])} major versions, averaging {stats['time_metrics']['releases_per_month']} releases per month" for p, stats in zip(df['project'], df.to_dict('records'))])}
    
    LATEST VERSIONS:
    ---------------------------------------------------------
    {chr(10).join([f"{p}: {stats['version_stats']['latest_version']}" for p, stats in zip(df['project'], df.to_dict('records'))])}
    
    =========================================================
    Images saved to '{output_dir}' directory
    =========================================================
    """
    
    # Save the report to a text file
    with open(f"{output_dir}/comparison_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"Comparison analysis complete! Report and visualizations saved to '{output_dir}' directory.")

    # Add this after all other visualizations
    # Create an integrated visualization combining key metrics
    plt.figure(figsize=(16, 18))  # Taller figure for three charts

    # Get consistent project ordering across all charts
    common_projects = []
    for project in all_project_releases.keys():
        if project in project_colors:
            common_projects.append(project)

    # 1. First subplot: Average Monthly Releases
    ax1 = plt.subplot(3, 1, 1)
    if true_monthly_averages:
        avg_df = pd.DataFrame(true_monthly_averages)
        avg_df = avg_df.set_index('project')
        
        # Ensure consistent project ordering
        if not avg_df.empty:
            available_projects = [p for p in common_projects if p in avg_df.index]
            avg_df = avg_df.reindex(available_projects)
            
            # Create bar chart
            avg_df['avg_releases'].plot(kind='bar', ax=ax1, color=[project_colors[p] for p in avg_df.index])
            
            # For the integrated view
            for i, (idx, row) in enumerate(avg_df.iterrows()):
                project = idx
                active_period = f"{row['first_release']} to {row['last_release']}"
                months_info = f"({row['months_active']} months)"
                
                # Only add text annotation if the bar is tall enough
                if row['avg_releases'] > 0.7:  # Only add text if bar is tall enough
                    ax1.annotate(
                        f"{active_period}\n{months_info}",
                        xy=(i, row['avg_releases'] / 2),  # Position in the middle of the bar
                        ha='center', va='center',  # Center alignment
                        fontsize=7,
                        color='white',  # White text for contrast
                        weight='bold',  # Make it bold for better readability
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.3)
                    )
        
        ax1.set_title(f'Average Monthly Releases During Active Period {timeframe_text}', fontsize=14)
        ax1.set_ylabel('Releases per Month')
        plt.xticks(rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Second subplot: Release Types Distribution
    ax2 = plt.subplot(3, 1, 2)
    types_data = []

    for project in common_projects:
        project_stats = next((s for s in stats_list if s['project'] == project), None)
        if project_stats:
            total = project_stats['total_releases']
            if total > 0:
                mandatory_pct = project_stats['release_types']['mandatory'] / total * 100
                optional_pct = project_stats['release_types']['optional'] / total * 100
                unknown_pct = project_stats['release_types']['unknown'] / total * 100
                
                types_data.append({
                    'project': project,
                    'Mandatory': mandatory_pct,
                    'Optional': optional_pct,
                    'Unknown': unknown_pct
                })

    if types_data:
        types_df = pd.DataFrame(types_data).set_index('project')
        types_df.plot(kind='bar', stacked=True, ax=ax2, 
                     color=['#ff6666', '#66b3ff', '#c2c2f0'])
        
        # Add percentage labels on bars
        for i, (idx, row) in enumerate(types_df.iterrows()):
            mandatory = row['Mandatory']
            optional = row['Optional']
            
            # Only label if significant percentage
            if mandatory > 10:
                ax2.annotate(f"{mandatory:.0f}%", 
                            xy=(i, mandatory/2), 
                            ha='center', va='center',
                            color='white', fontweight='bold')
            
            if optional > 10:
                ax2.annotate(f"{optional:.0f}%", 
                            xy=(i, mandatory + optional/2), 
                            ha='center', va='center',
                            color='white', fontweight='bold')
        
        ax2.set_title(f'Release Types Distribution {timeframe_text} (Lower Mandatory % = More Stable)', fontsize=14)
        ax2.set_ylabel('Percentage')
        ax2.set_ylim(0, 100)  # Force scale to 0-100%
        plt.xticks(rotation=45, ha='right')
        ax2.legend(title='Release Type')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Third subplot: Average Days Between Releases
    ax3 = plt.subplot(3, 1, 3)
    freq_data = []

    for project in common_projects:
        project_stats = next((s for s in stats_list if s['project'] == project), None)
        if project_stats:
            avg_days = project_stats['time_metrics']['avg_days_between_releases']
            freq_data.append({
                'project': project,
                'avg_days': avg_days
            })

    if freq_data:
        freq_df = pd.DataFrame(freq_data).set_index('project')
        bars = freq_df['avg_days'].plot(kind='bar', ax=ax3, color=[project_colors[p] for p in freq_df.index])
        
        # Add value labels on top of bars
        for i, (idx, row) in enumerate(freq_df.iterrows()):
            ax3.annotate(f"{row['avg_days']:.1f} days", 
                        xy=(i, row['avg_days']),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax3.set_title(f'Average Days Between Releases {timeframe_text}', fontsize=14)
        ax3.set_ylabel('Days')
        plt.xticks(rotation=45, ha='right')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{output_dir}/integrated_stability_analysis.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare GitHub release statistics across projects')
    
    # Update default paths to be more flexible
    parser.add_argument('--celestia-stats', default=None, help='Path to Celestia statistics JSON')
    parser.add_argument('--arbitrum-stats', default=None, help='Path to Arbitrum statistics JSON')
    parser.add_argument('--eigenda-stats', default=None, help='Path to EigenDA statistics JSON')
    parser.add_argument('--avalanche-stats', default=None, help='Path to Avalanche statistics JSON')
    parser.add_argument('--optimism-stats', default=None, help='Path to Optimism statistics JSON')
    parser.add_argument('--op-succinct-stats', default=None, help='Path to OP Succinct statistics JSON')
    parser.add_argument('--output-dir', default='comparison_analysis', help='Directory to save comparison analysis')
    args = parser.parse_args()
    
    # Look for statistics files in different possible locations
    def find_stats_file(project, provided_path):
        if provided_path and os.path.exists(provided_path):
            return provided_path
            
        # Try different common locations
        possible_paths = [
            f"{project.lower()}_analysis/release_statistics.json",
            f"release_analysis/{project.lower()}_release_statistics.json",
            f"release_analysis/release_statistics.json"  # If only one project was analyzed
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found statistics for {project} at {path}")
                return path
                
        return None
    
    celestia_path = find_stats_file("Celestia", args.celestia_stats)
    arbitrum_path = find_stats_file("Arbitrum", args.arbitrum_stats)
    eigenda_path = find_stats_file("EigenDA", args.eigenda_stats)
    avalanche_path = find_stats_file("Avalanche", args.avalanche_stats)
    optimism_path = find_stats_file("Optimism", args.optimism_stats)
    op_succinct_path = find_stats_file("OPSuccinct", args.op_succinct_stats)
    
    # Load statistics for each project
    stats_list = []
    
    if celestia_path:
        stats = load_stats('Celestia', celestia_path)
        if stats:
            stats_list.append(stats)
    
    if arbitrum_path:
        stats = load_stats('Arbitrum', arbitrum_path)
        if stats:
            stats_list.append(stats)
    
    if eigenda_path:
        stats = load_stats('EigenDA', eigenda_path)
        if stats:
            stats_list.append(stats)
    
    if avalanche_path:
        stats = load_stats('Avalanche', avalanche_path)
        if stats:
            stats_list.append(stats)
    
    if optimism_path:
        stats = load_stats('Optimism', optimism_path)
        if stats:
            stats_list.append(stats)
    
    if op_succinct_path:
        stats = load_stats('OP Succinct', op_succinct_path)
        if stats:
            stats_list.append(stats)
    
    if not stats_list:
        print("No statistics files were found. Please run the analysis for each project first.")
        print("Example commands:")
        print("  ./celestia_release_analytics.py celestia-node_releases.json --output-dir celestia_analysis")
        print("  ./celestia_release_analytics.py arbitrum_releases.json --output-dir arbitrum_analysis")
        print("  ./celestia_release_analytics.py eigenda_releases.json --output-dir eigenda_analysis")
        print("  ./celestia_release_analytics.py avalanche_releases.json --output-dir avalanche_analysis")
        sys.exit(1)
    
    # Generate comparison report
    create_comparison_report(stats_list, args.output_dir)

if __name__ == "__main__":
    main()
