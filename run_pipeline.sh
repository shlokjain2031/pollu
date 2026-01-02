#!/bin/bash
# Complete pipeline runner: patliputra -> nalanda/target -> nalanda/training
# Sequentially runs data fetching, target assembly, and training feature assembly
# with optional parallelization support.
#
# Usage:
#   ./run_pipeline.sh \
#     --city mumbai \
#     --start-date 2023-01-01 --end-date 2023-01-31 \
#     --openaq-token <openaq-token> \
#     --bbox "72.7763,18.8939,72.9797,19.2701" \
#     --ee-project your-ee-project \
#     --hourly \
#     --parallel --workers 4
#
# Required for patliputra: city, start-date, end-date, openaq-token, ee-project, hourly
# Required for target: start-date, end-date, openaq-token, bbox, parallel, workers
# Required for training: start-date, end-date, parallel, workers

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command-line arguments
PARALLEL_FLAG=""
WORKERS_FLAG=""
EE_PROJECT=""
CITY=""
START_DATE=""
END_DATE=""
OPENAQ_TOKEN=""
BBOX=""
HOURLY_FLAG=""
VERBOSE_FLAG="-v"  # Always verbose

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel|-p)
            PARALLEL_FLAG="--parallel"
            shift
            ;;
        --workers|-w)
            WORKERS_FLAG="--workers $2"
            shift 2
            ;;
        --ee-project|-e)
            EE_PROJECT="$2"
            shift 2
            ;;
        --city|-c)
            CITY="$2"
            shift 2
            ;;
        --start-date|-s)
            START_DATE="$2"
            shift 2
            ;;
        --end-date|-ed)
            END_DATE="$2"
            shift 2
            ;;
        --openaq-token|-ot)
            OPENAQ_TOKEN="$2"
            shift 2
            ;;
        --bbox|-b)
            BBOX="$2"
            shift 2
            ;;
        --hourly|-h)
            HOURLY_FLAG="--hourly"
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            shift
            ;;
    esac
done

# Trim whitespace
PARALLEL_FLAG=$(echo $PARALLEL_FLAG | xargs)
WORKERS_FLAG=$(echo $WORKERS_FLAG | xargs)
EE_PROJECT=$(echo $EE_PROJECT | xargs)
CITY=$(echo $CITY | xargs)
START_DATE=$(echo $START_DATE | xargs)
END_DATE=$(echo $END_DATE | xargs)
OPENAQ_TOKEN=$(echo $OPENAQ_TOKEN | xargs)
BBOX=$(echo $BBOX | xargs)
HOURLY_FLAG=$(echo $HOURLY_FLAG | xargs)

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Pollu Complete Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Verify Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found. Please activate your virtual environment.${NC}"
    exit 1
fi

echo -e "${YELLOW}Environment:${NC}"
echo "  Python: $(python --version)"
echo "  Current directory: $(pwd)"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  City: ${CITY:-<not set>}"
echo "  Start date: ${START_DATE:-<not set>}"
echo "  End date: ${END_DATE:-<not set>}"
echo "  OpenAQ token: ${OPENAQ_TOKEN:0:10}***** (${#OPENAQ_TOKEN} chars)"
echo "  Bounding box: ${BBOX:-<not set>}"
echo "  EE Project: ${EE_PROJECT:-<not set>}"
echo "  Hourly: ${HOURLY_FLAG:-disabled}"
echo "  Parallel: ${PARALLEL_FLAG:-disabled}"
if [[ -n "$WORKERS_FLAG" ]]; then
    echo "  Workers: $(echo $WORKERS_FLAG | awk '{print $2}')"
fi
echo ""

# ============================================================================
# Step 1: Patliputra - Data fetching
# ============================================================================
echo -e "${BLUE}Step 1: Data Fetching (patliputra/pipeline.py)${NC}"
echo -e "${BLUE}───────────────────────────────────────────────${NC}"

# Validate required arguments for patliputra
PATLIPUTRA_REQUIRED=("CITY" "START_DATE" "END_DATE" "OPENAQ_TOKEN" "EE_PROJECT")
MISSING_PATLIPUTRA=()
for arg in "${PATLIPUTRA_REQUIRED[@]}"; do
    if [[ -z "${!arg}" ]]; then
        MISSING_PATLIPUTRA+=("--${arg,,}")
    fi
done

if [[ ${#MISSING_PATLIPUTRA[@]} -gt 0 ]]; then
    echo -e "${RED}✗ Missing required arguments for patliputra:${NC}"
    for arg in "${MISSING_PATLIPUTRA[@]}"; do
        echo "  - $arg"
    done
    exit 1
fi

if [[ -f "patliputra/pipeline.py" ]]; then
    PATLIPUTRA_CMD="python patliputra/pipeline.py $VERBOSE_FLAG"
    PATLIPUTRA_CMD="$PATLIPUTRA_CMD --city $CITY"
    PATLIPUTRA_CMD="$PATLIPUTRA_CMD --start-date $START_DATE"
    PATLIPUTRA_CMD="$PATLIPUTRA_CMD --end-date $END_DATE"
    PATLIPUTRA_CMD="$PATLIPUTRA_CMD --openaq-token $OPENAQ_TOKEN"
    PATLIPUTRA_CMD="$PATLIPUTRA_CMD --ee-project $EE_PROJECT"
    if [[ -n "$HOURLY_FLAG" ]]; then
        PATLIPUTRA_CMD="$PATLIPUTRA_CMD $HOURLY_FLAG"
    fi
    
    echo "Command: $PATLIPUTRA_CMD"
    echo ""
    
    if eval $PATLIPUTRA_CMD; then
        echo -e "${GREEN}✓ Data fetching completed${NC}"
    else
        echo -e "${RED}✗ Data fetching failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⊘ patliputra/pipeline.py not found, skipping...${NC}"
fi

echo ""

# ============================================================================
# Step 2: Nalanda Target Assembly
# ============================================================================
echo -e "${BLUE}Step 2: Target Assembly (nalanda/target_cli.py)${NC}"
echo -e "${BLUE}───────────────────────────────────────────────${NC}"

# Validate required arguments for target
TARGET_REQUIRED=("START_DATE" "END_DATE" "OPENAQ_TOKEN" "BBOX")
MISSING_TARGET=()
for arg in "${TARGET_REQUIRED[@]}"; do
    if [[ -z "${!arg}" ]]; then
        MISSING_TARGET+=("--${arg,,}")
    fi
done

if [[ ${#MISSING_TARGET[@]} -gt 0 ]]; then
    echo -e "${RED}✗ Missing required arguments for target assembly:${NC}"
    for arg in "${MISSING_TARGET[@]}"; do
        echo "  - $arg"
    done
    exit 1
fi

TARGET_CMD="python -m nalanda.target_cli $VERBOSE_FLAG"
TARGET_CMD="$TARGET_CMD --start-date $START_DATE"
TARGET_CMD="$TARGET_CMD --end-date $END_DATE"
TARGET_CMD="$TARGET_CMD --openaq-token $OPENAQ_TOKEN"
TARGET_CMD="$TARGET_CMD --bbox \"$BBOX\""
if [[ -n "$PARALLEL_FLAG" ]]; then
    TARGET_CMD="$TARGET_CMD $PARALLEL_FLAG"
fi
if [[ -n "$WORKERS_FLAG" ]]; then
    TARGET_CMD="$TARGET_CMD $WORKERS_FLAG"
fi

echo "Command: $TARGET_CMD"
echo ""

if eval $TARGET_CMD; then
    echo -e "${GREEN}✓ Target assembly completed${NC}"
else
    echo -e "${RED}✗ Target assembly failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Step 3: Nalanda Training Assembly
# ============================================================================
echo -e "${BLUE}Step 3: Training Assembly (nalanda/cli.py assemble)${NC}"
echo -e "${BLUE}──────────────────────────────────────────────────${NC}"

# Validate required arguments for training
TRAINING_REQUIRED=("START_DATE" "END_DATE")
MISSING_TRAINING=()
for arg in "${TRAINING_REQUIRED[@]}"; do
    if [[ -z "${!arg}" ]]; then
        MISSING_TRAINING+=("--${arg,,}")
    fi
done

if [[ ${#MISSING_TRAINING[@]} -gt 0 ]]; then
    echo -e "${RED}✗ Missing required arguments for training assembly:${NC}"
    for arg in "${MISSING_TRAINING[@]}"; do
        echo "  - $arg"
    done
    exit 1
fi

TRAINING_CMD="python -m nalanda.cli assemble $VERBOSE_FLAG"
TRAINING_CMD="$TRAINING_CMD --start-date $START_DATE"
TRAINING_CMD="$TRAINING_CMD --end-date $END_DATE"
if [[ -n "$PARALLEL_FLAG" ]]; then
    TRAINING_CMD="$TRAINING_CMD $PARALLEL_FLAG"
fi
if [[ -n "$WORKERS_FLAG" ]]; then
    TRAINING_CMD="$TRAINING_CMD $WORKERS_FLAG"
fi

echo "Command: $TRAINING_CMD"
echo ""

if eval $TRAINING_CMD; then
    echo -e "${GREEN}✓ Training assembly completed${NC}"
else
    echo -e "${RED}✗ Training assembly failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Pipeline completed successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  ✓ Data fetching (patliputra)"
echo "  ✓ Target assembly (nalanda/target_cli.py)"
echo "  ✓ Training assembly (nalanda/cli.py)"
echo ""
echo -e "${YELLOW}Output locations:${NC}"
echo "  - Targets: artifacts/v1/training/{date}/target.parquet"
echo "  - Training: artifacts/v1/training/{date}/training.parquet"
echo ""
