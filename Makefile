# GitHub Release Analysis Makefile

# Default parameters
MONTHS := 6
DEBUG := false
RAW_HTML := false
RATE_LIMIT := true

# Scripts directory
SCRIPTS_DIR := scripts

# JSON output directory
JSON_DIR := jsons

# Analysis base directory
ANALYSIS_BASE_DIR := analysis

# Repositories to analyze
REPOS := celestia arbitrum eigenda avalanche optimism op_succinct

# Repository URLs
CELESTIA_URL := github.com/celestiaorg/celestia-node/releases
ARBITRUM_URL := github.com/OffchainLabs/nitro/releases
EIGENDA_URL := github.com/Layr-Labs/eigenda/releases
AVALANCHE_URL := github.com/ava-labs/avalanchego/releases
OPTIMISM_URL := github.com/ethereum-optimism/optimism/releases
OP_SUCCINCT_URL := github.com/succinctlabs/op-succinct/releases

# Output files
CELESTIA_JSON := $(JSON_DIR)/celestia-node_releases.json
ARBITRUM_JSON := $(JSON_DIR)/arbitrum_releases.json
EIGENDA_JSON := $(JSON_DIR)/eigenda_releases.json
AVALANCHE_JSON := $(JSON_DIR)/avalanche_releases.json
OPTIMISM_JSON := $(JSON_DIR)/optimism_releases.json
OP_SUCCINCT_JSON := $(JSON_DIR)/op_succinct_releases.json

# Analysis directories
CELESTIA_ANALYSIS := $(ANALYSIS_BASE_DIR)/celestia_analysis
ARBITRUM_ANALYSIS := $(ANALYSIS_BASE_DIR)/arbitrum_analysis
EIGENDA_ANALYSIS := $(ANALYSIS_BASE_DIR)/eigenda_analysis
AVALANCHE_ANALYSIS := $(ANALYSIS_BASE_DIR)/avalanche_analysis
OPTIMISM_ANALYSIS := $(ANALYSIS_BASE_DIR)/optimism_analysis
OP_SUCCINCT_ANALYSIS := $(ANALYSIS_BASE_DIR)/op_succinct_analysis
COMPARISON_DIR := comparison_analysis

# Default target to run everything
all: extract analyze compare

# Clean all generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(CELESTIA_JSON) $(ARBITRUM_JSON) $(EIGENDA_JSON) $(AVALANCHE_JSON) $(OPTIMISM_JSON) $(OP_SUCCINCT_JSON)
	rm -rf $(CELESTIA_ANALYSIS) $(ARBITRUM_ANALYSIS) $(EIGENDA_ANALYSIS) $(AVALANCHE_ANALYSIS) $(OPTIMISM_ANALYSIS) $(OP_SUCCINCT_ANALYSIS) $(COMPARISON_DIR)

# Ensure directories exist
setup:
	@echo "Setting up directory structure..."
	mkdir -p $(JSON_DIR) $(ANALYSIS_BASE_DIR) $(COMPARISON_DIR)

# Target to extract data for all repositories
extract: setup extract_celestia extract_arbitrum extract_eigenda extract_avalanche extract_optimism extract_op_succinct

# Targets to extract data for each repository
extract_celestia:
	@echo "Extracting Celestia release data..."
	python $(SCRIPTS_DIR)/github_release_extractor.py $(CELESTIA_URL) --output $(CELESTIA_JSON) --months $(MONTHS) $(if $(filter true,$(DEBUG)),--debug,) $(if $(filter true,$(RAW_HTML)),--raw-html,) $(if $(filter true,$(RATE_LIMIT)),--rate-limit,)

extract_arbitrum:
	@echo "Extracting Arbitrum release data..."
	python $(SCRIPTS_DIR)/github_release_extractor.py $(ARBITRUM_URL) --output $(ARBITRUM_JSON) --months $(MONTHS) $(if $(filter true,$(DEBUG)),--debug,) $(if $(filter true,$(RAW_HTML)),--raw-html,) $(if $(filter true,$(RATE_LIMIT)),--rate-limit,)

extract_eigenda:
	@echo "Extracting EigenDA release data..."
	python $(SCRIPTS_DIR)/github_release_extractor.py $(EIGENDA_URL) --output $(EIGENDA_JSON) --months $(MONTHS) $(if $(filter true,$(DEBUG)),--debug,) $(if $(filter true,$(RAW_HTML)),--raw-html,) $(if $(filter true,$(RATE_LIMIT)),--rate-limit,)

extract_avalanche:
	@echo "Extracting Avalanche release data..."
	python $(SCRIPTS_DIR)/github_release_extractor.py $(AVALANCHE_URL) --output $(AVALANCHE_JSON) --months $(MONTHS) $(if $(filter true,$(DEBUG)),--debug,) $(if $(filter true,$(RAW_HTML)),--raw-html,) $(if $(filter true,$(RATE_LIMIT)),--rate-limit,)

extract_optimism:
	@echo "Extracting Optimism release data..."
	python $(SCRIPTS_DIR)/github_release_extractor.py $(OPTIMISM_URL) --output $(OPTIMISM_JSON) --months $(MONTHS) $(if $(filter true,$(DEBUG)),--debug,) $(if $(filter true,$(RAW_HTML)),--raw-html,) $(if $(filter true,$(RATE_LIMIT)),--rate-limit,)

extract_op_succinct:
	@echo "Extracting OP Succinct release data..."
	python $(SCRIPTS_DIR)/github_release_extractor.py $(OP_SUCCINCT_URL) --output $(OP_SUCCINCT_JSON) --months $(MONTHS) $(if $(filter true,$(DEBUG)),--debug,) $(if $(filter true,$(RAW_HTML)),--raw-html,) $(if $(filter true,$(RATE_LIMIT)),--rate-limit,)

# Target to analyze all repositories
analyze: extract analyze_celestia analyze_arbitrum analyze_eigenda analyze_avalanche analyze_optimism analyze_op_succinct

# Targets to analyze each repository
analyze_celestia: extract_celestia
	@echo "Analyzing Celestia releases..."
	python $(SCRIPTS_DIR)/analytic_releases.py $(CELESTIA_JSON) --output-dir $(CELESTIA_ANALYSIS)

analyze_arbitrum: extract_arbitrum
	@echo "Analyzing Arbitrum releases..."
	python $(SCRIPTS_DIR)/analytic_releases.py $(ARBITRUM_JSON) --output-dir $(ARBITRUM_ANALYSIS)

analyze_eigenda: extract_eigenda
	@echo "Analyzing EigenDA releases..."
	python $(SCRIPTS_DIR)/analytic_releases.py $(EIGENDA_JSON) --output-dir $(EIGENDA_ANALYSIS)

analyze_avalanche: extract_avalanche
	@echo "Analyzing Avalanche releases..."
	python $(SCRIPTS_DIR)/analytic_releases.py $(AVALANCHE_JSON) --output-dir $(AVALANCHE_ANALYSIS)

analyze_optimism: extract_optimism
	@echo "Analyzing Optimism releases..."
	python $(SCRIPTS_DIR)/analytic_releases.py $(OPTIMISM_JSON) --output-dir $(OPTIMISM_ANALYSIS)

analyze_op_succinct: extract_op_succinct
	@echo "Analyzing OP Succinct releases..."
	python $(SCRIPTS_DIR)/analytic_releases.py $(OP_SUCCINCT_JSON) --output-dir $(OP_SUCCINCT_ANALYSIS)

# Target to generate comparison report
compare:
	@echo "Generating comparison report..."
	python $(SCRIPTS_DIR)/compare_releases.py \
		--celestia-stats $(CELESTIA_ANALYSIS)/release_statistics.json \
		--arbitrum-stats $(ARBITRUM_ANALYSIS)/release_statistics.json \
		--eigenda-stats $(EIGENDA_ANALYSIS)/release_statistics.json \
		--avalanche-stats $(AVALANCHE_ANALYSIS)/release_statistics.json \
		--optimism-stats $(OPTIMISM_ANALYSIS)/release_statistics.json \
		--op-succinct-stats $(OP_SUCCINCT_ANALYSIS)/release_statistics.json \
		--output-dir $(COMPARISON_DIR)

# Help target
help:
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all           Run the entire pipeline (extract, analyze, compare)"
	@echo "  extract       Extract data from all repositories"
	@echo "  analyze       Analyze data from all repositories"
	@echo "  compare       Generate comparison report"
	@echo "  clean         Clean all generated files"
	@echo "  setup         Create necessary directories"
	@echo ""
	@echo "Options:"
	@echo "  MONTHS=n      Look back n months (default: 24)"
	@echo "  DEBUG=true    Enable debug output"
	@echo "  RAW_HTML=true Save raw HTML for debugging"
	@echo "  RATE_LIMIT=false Disable rate limiting"
	@echo ""
	@echo "Examples:"
	@echo "  make all MONTHS=6             Run everything with 6 months lookback"
	@echo "  make extract_celestia MONTHS=3 Extract only Celestia data for 3 months"
	@echo "  make clean                    Clean all generated files"

.PHONY: all clean setup extract analyze compare extract_celestia extract_arbitrum extract_eigenda extract_avalanche extract_optimism extract_op_succinct analyze_celestia analyze_arbitrum analyze_eigenda analyze_avalanche analyze_optimism analyze_op_succinct help 