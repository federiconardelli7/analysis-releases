# GitHub Release Analysis Makefile

# Default parameters
MONTHS := 24
DEBUG := false
RAW_HTML := false
RATE_LIMIT := true
PROJECTS := celestia arbitrum eigenda avalanche optimism op_succinct

# Scripts directory
SCRIPTS_DIR := scripts

# JSON output directory
JSON_DIR := jsons

# Analysis base directory
ANALYSIS_BASE_DIR := analysis

# All available repositories
ALL_REPOS := celestia arbitrum eigenda avalanche optimism op_succinct cosmos

# Repository URLs
CELESTIA_URL := github.com/celestiaorg/celestia-node/releases
ARBITRUM_URL := github.com/OffchainLabs/nitro/releases
EIGENDA_URL := github.com/Layr-Labs/eigenda/releases
AVALANCHE_URL := github.com/ava-labs/avalanchego/releases
OPTIMISM_URL := github.com/ethereum-optimism/optimism/releases
OP_SUCCINCT_URL := github.com/succinctlabs/op-succinct/releases
COSMOS_URL := github.com/cosmos/cosmos-sdk/releases

# Output files
CELESTIA_JSON := $(JSON_DIR)/celestia-node_releases.json
ARBITRUM_JSON := $(JSON_DIR)/arbitrum_releases.json
EIGENDA_JSON := $(JSON_DIR)/eigenda_releases.json
AVALANCHE_JSON := $(JSON_DIR)/avalanche_releases.json
OPTIMISM_JSON := $(JSON_DIR)/optimism_releases.json
OP_SUCCINCT_JSON := $(JSON_DIR)/op_succinct_releases.json
COSMOS_JSON := $(JSON_DIR)/cosmos_releases.json

# Analysis directories
CELESTIA_ANALYSIS := $(ANALYSIS_BASE_DIR)/celestia_analysis
ARBITRUM_ANALYSIS := $(ANALYSIS_BASE_DIR)/arbitrum_analysis
EIGENDA_ANALYSIS := $(ANALYSIS_BASE_DIR)/eigenda_analysis
AVALANCHE_ANALYSIS := $(ANALYSIS_BASE_DIR)/avalanche_analysis
OPTIMISM_ANALYSIS := $(ANALYSIS_BASE_DIR)/optimism_analysis
OP_SUCCINCT_ANALYSIS := $(ANALYSIS_BASE_DIR)/op_succinct_analysis
COSMOS_ANALYSIS := $(ANALYSIS_BASE_DIR)/cosmos_analysis
COMPARISON_DIR := comparison_analysis

# Default target to run everything
all: extract analyze compare

# Clean all generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(CELESTIA_JSON) $(ARBITRUM_JSON) $(EIGENDA_JSON) $(AVALANCHE_JSON) $(OPTIMISM_JSON) $(OP_SUCCINCT_JSON) $(COSMOS_JSON)
	rm -rf $(CELESTIA_ANALYSIS) $(ARBITRUM_ANALYSIS) $(EIGENDA_ANALYSIS) $(AVALANCHE_ANALYSIS) $(OPTIMISM_ANALYSIS) $(OP_SUCCINCT_ANALYSIS) $(COSMOS_ANALYSIS) $(COMPARISON_DIR)

# Ensure directories exist
setup:
	@echo "Setting up directory structure..."
	mkdir -p $(JSON_DIR) $(ANALYSIS_BASE_DIR) $(COMPARISON_DIR)

# Target to extract data for selected repositories
extract: setup $(foreach repo,$(PROJECTS),extract_$(repo))

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

extract_cosmos:
	@echo "Extracting Cosmos release data..."
	python $(SCRIPTS_DIR)/github_release_extractor.py $(COSMOS_URL) --output $(COSMOS_JSON) --months $(MONTHS) $(if $(filter true,$(DEBUG)),--debug,) $(if $(filter true,$(RAW_HTML)),--raw-html,) $(if $(filter true,$(RATE_LIMIT)),--rate-limit,)

# Target to analyze selected repositories
analyze: extract $(foreach repo,$(PROJECTS),analyze_$(repo))

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

analyze_cosmos: extract_cosmos
	@echo "Analyzing Cosmos releases..."
	python $(SCRIPTS_DIR)/analytic_releases.py $(COSMOS_JSON) --output-dir $(COSMOS_ANALYSIS)

# Target to generate comparison report based on selected projects
compare:
	@echo "Generating comparison report..."
	python $(SCRIPTS_DIR)/compare_releases.py \
		--months $(MONTHS) \
		--output-dir $(COMPARISON_DIR) \
		--projects $(foreach proj,$(PROJECTS),$(proj):$(ANALYSIS_BASE_DIR)/$(proj)_analysis/release_statistics.json)

# Help target
help:
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all           Run the entire pipeline (extract, analyze, compare)"
	@echo "  extract       Extract data from selected repositories"
	@echo "  analyze       Analyze data from selected repositories"
	@echo "  compare       Generate comparison report"
	@echo "  clean         Clean all generated files"
	@echo "  setup         Create necessary directories"
	@echo ""
	@echo "Options:"
	@echo "  MONTHS=n      Look back n months (default: 6)"
	@echo "  PROJECTS='p1 p2 p3' Select which projects to analyze (default: celestia arbitrum eigenda avalanche optimism op_succinct)"
	@echo "  DEBUG=true    Enable debug output"
	@echo "  RAW_HTML=true Save raw HTML for debugging"
	@echo "  RATE_LIMIT=false Disable rate limiting"
	@echo ""
	@echo "Available projects: $(ALL_REPOS)"
	@echo ""
	@echo "Examples:"
	@echo "  make all MONTHS=6                      Run everything with 6 months lookback"
	@echo "  make PROJECTS='celestia cosmos'        Analyze only Celestia and Cosmos"
	@echo "  make extract_celestia MONTHS=3         Extract only Celestia data for 3 months"
	@echo "  make clean                             Clean all generated files"

.PHONY: all clean setup extract analyze compare $(foreach repo,$(ALL_REPOS),extract_$(repo) analyze_$(repo)) help