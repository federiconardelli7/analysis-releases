#!/usr/bin/env bash

# Default values
MONTHS=12
PROJECT="all"
ACTION="all"
PROJECTS="celestia arbitrum eigenda avalanche optimism op_succinct"

# First, check if directories exist and create them if needed
mkdir -p scripts jsons analysis comparison_analysis

# Check if script files are in the right place
if [ ! -f "scripts/github_release_extractor.py" ] && [ -f "github_release_extractor.py" ]; then
    echo "Moving scripts to scripts/ directory..."
    mv github_release_extractor.py scripts/
    mv analytic_releases.py scripts/
    mv compare_releases.py scripts/
fi

# Help function
function show_help {
    echo "Usage: ./analyze_releases.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --months N    Look back N months (default: 12)"
    echo "  -p, --project X   Analyze only project X (default: all)"
    echo "                    Valid projects: celestia, arbitrum, eigenda, avalanche, optimism, op_succinct, cosmos"
    echo "  -a, --action Y    Perform only action Y (default: all)"
    echo "                    Valid actions: extract, analyze, compare"
    echo "  -l, --list L      Comma-separated list of projects to analyze"
    echo "  -d, --debug       Enable debug mode"
    echo "  -r, --raw-html    Save raw HTML for debugging"
    echo "  -n, --no-rate-limit Disable rate limiting"
    echo "  -c, --clean       Clean all generated files before starting"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./analyze_releases.sh -m 6                 Run all with 6 months lookback"
    echo "  ./analyze_releases.sh -p celestia -a extract Extract only Celestia data"
    echo "  ./analyze_releases.sh -l 'celestia,cosmos' Analyze only Celestia and Cosmos"
    echo "  ./analyze_releases.sh -c -m 3              Clean and run with 3 months lookback"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--months)
            MONTHS="$2"
            shift 2
            ;;
        -p|--project)
            PROJECT="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -l|--list)
            PROJECTS="${2//,/ }" # Convert comma-separated list to space-separated
            PROJECT="custom"
            shift 2
            ;;
        -d|--debug)
            DEBUG="true"
            shift
            ;;
        -r|--raw-html)
            RAW_HTML="true"
            shift
            ;;
        -n|--no-rate-limit)
            RATE_LIMIT="false"
            shift
            ;;
        -c|--clean)
            CLEAN="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build make command
MAKE_CMD="make"

# Add clean if requested
if [[ "$CLEAN" == "true" ]]; then
    MAKE_CMD="$MAKE_CMD clean setup"
else
    MAKE_CMD="$MAKE_CMD setup"
fi

# Add project and action
if [[ "$PROJECT" == "all" ]]; then
    if [[ "$ACTION" == "all" ]]; then
        MAKE_CMD="$MAKE_CMD all"
    else
        MAKE_CMD="$MAKE_CMD $ACTION"
    fi
elif [[ "$PROJECT" == "custom" ]]; then
    # For custom list of projects
    MAKE_CMD="$MAKE_CMD PROJECTS='$PROJECTS'"
    if [[ "$ACTION" == "all" ]]; then
        MAKE_CMD="$MAKE_CMD all"
    else
        MAKE_CMD="$MAKE_CMD $ACTION"
    fi
else
    if [[ "$ACTION" == "all" ]]; then
        MAKE_CMD="$MAKE_CMD extract_$PROJECT analyze_$PROJECT"
        # Only add compare if we're not doing a single project
        if [[ "$PROJECT" == "all" ]]; then
            MAKE_CMD="$MAKE_CMD compare"
        fi
    else
        MAKE_CMD="$MAKE_CMD ${ACTION}_$PROJECT"
    fi
fi

# Add options
MAKE_CMD="$MAKE_CMD MONTHS=$MONTHS"
[[ "$DEBUG" == "true" ]] && MAKE_CMD="$MAKE_CMD DEBUG=true"
[[ "$RAW_HTML" == "true" ]] && MAKE_CMD="$MAKE_CMD RAW_HTML=true"
[[ "$RATE_LIMIT" == "false" ]] && MAKE_CMD="$MAKE_CMD RATE_LIMIT=false"

# Run the command
echo "Running: $MAKE_CMD"
$MAKE_CMD 