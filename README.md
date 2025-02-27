# Layer 2 Release Analysis Tools

This project provides tools to analyze and compare GitHub releases from various Layer 2 solutions and related projects against Avalanche.

## Projects Analyzed

- Celestia (`celestiaorg/celestia-node`)
- Arbitrum (`OffchainLabs/nitro`)
- EigenDA (`Layr-Labs/eigenda`)
- Avalanche (`ava-labs/avalanchego`)
- Optimism (`ethereum-optimism/optimism`)
- OP Succinct (`succinctlabs/op-succinct`)

## Directory Structure

```
project-root/
├── Makefile                   # Main makefile
├── README.md                  # This file
├── github-extractor-env/      # Python environment
├── scripts/                   # Python scripts folder
│   ├── github_release_extractor.py
│   ├── analytic_releases.py
│   └── compare_releases.py
├── jsons/                     # Output JSON files
├── analysis/                  # Individual analysis folders
└── comparison_analysis/       # Comparison results
```

## Setup

1. Ensure Python 3.7+ is installed
2. Install required dependencies:

bash
pip install pandas matplotlib seaborn requests

3. Make the shell script executable:

bash
chmod +x analyze_releases.sh

## Environment Setup

### Python Environment Setup

```bash
# Create a virtual environment
python -m venv github-extractor-env

# Activate the environment
# On Windows:
github-extractor-env\Scripts\activate
# On macOS/Linux:
source github-extractor-env/bin/activate

# Install required dependencies
pip install pandas matplotlib seaborn requests
```

## Running the Analysis

### Using the shell script (recommended)

The simplest way to run the analysis is using the provided shell script:
bash:README.md
./analyze_releases.sh


This will extract data for all projects for the last 12 months, analyze each project, and create a comparison report.

### Options

You can customize the analysis with various options:

bash:README.md
Look back 6 months instead of 12
./analyze_releases.sh --months 6
Analyze only Celestia
./analyze_releases.sh --project celestia
Only extract data without analyzing
./analyze_releases.sh --action extract
Clean previous results and run with debug info
./analyze_releases.sh --clean --debug
Show all available options
./analyze_releases.sh --help

### Using the Makefile directly
### SUGGESTED
You can also use the Makefile directly if you prefer: 

bash:README.md
Run everything with default settings
make
Extract data for all projects with 6 months lookback
make extract MONTHS=6
Analyze only Celestia
make analyze_celestia
Only generate comparison report
make compare
Clean all generated files
make clean
Show help
make help

## Output

The analysis generates several outputs:

1. JSON files with raw release data for each project in the `jsons/` directory
2. Analysis directories for each project in the `analysis/` directory containing:
   - Visualizations (PNG images)
   - Statistics (JSON)
   - Reports (text)
3. A comparison directory with:
   - Cross-project visualizations
   - Comparison report

### Key Visualizations

- **Per-Project Visualizations**: Each project gets its own set of analysis charts
  - Monthly release frequency
  - Release type distribution (Mandatory/Optional/Unknown)
  - Pre-release vs Stable distribution
  - Network distribution (for multi-network projects)
  - Cumulative releases over time
  - Days between releases
  - Version distribution
  - Release timeline by network

- **Comparison Visualizations**: Cross-project analysis charts
  - Total releases by project
  - Average days between releases
  - Release types distribution comparison
  - Stability comparison (Stable vs Pre-release)
  - Releases per month comparison
  - Cumulative releases by project over time
  - Unified release timeline for all projects

## Customizing Projects

To add or modify projects:

1. Edit the Makefile to add the new repository URL and output paths
2. Update the `scripts/github_release_extractor.py` script if the new repository uses unique version naming patterns
3. Update the `scripts/analytic_releases.py` script to recognize network/version patterns for the new repository

## Scheduling Regular Updates

You can set up a cron job to run this analysis regularly:

```bash
# Run the analysis monthly
0 0 1 * * cd /path/to/project && ./analyze_releases.sh --months 12 > /path/to/logs/analysis_$(date +\%Y\%m\%d).log 2>&1
```

## How to Use These Files

1. Save these three files in your project directory:
   - `Makefile`
   - `analyze_releases.sh` (Make it executable with `chmod +x analyze_releases.sh`)
   - `README.md`

2. Now you can run the analysis in various ways:

