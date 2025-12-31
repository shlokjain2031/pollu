# Artifact Consumer Test

Generic validation test that acts as a downstream consumer of any data stream (landsat8, s5p, evi, etc.).

## Purpose

Validates that produced artifacts:
- Are registered correctly in the manifest
- Have correct schema (columns)
- Have expected row counts (match grid size)
- Have valid data ranges
- Follow deterministic row ordering (match canonical grid)
- Are readable and complete

## Usage

```bash
# Test landsat artifacts (all dates in manifest)
PYTHONPATH=. python3 tests/test_artifact_consumer.py \
  --artifact-type landsat_signals \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet

# Test specific dates only
PYTHONPATH=. python3 tests/test_artifact_consumer.py \
  --artifact-type landsat_signals \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --dates 2018-01-01 2018-01-17

# Test with custom manifest (e.g., test outputs)
PYTHONPATH=. python3 tests/test_artifact_consumer.py \
  --artifact-type landsat_signals \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet \
  --manifest-db test_output/landsat8_full/test_manifest.db \
  --dates 2018-01-01 2018-01-17

# Test other artifact types (when available)
PYTHONPATH=. python3 tests/test_artifact_consumer.py \
  --artifact-type s5p_no2_signals \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet

PYTHONPATH=. python3 tests/test_artifact_consumer.py \
  --artifact-type sentinel_evi_signals \
  --grid-artifact artifacts/v1/grids/mumbai/grid.parquet
```

## What It Checks

1. **Manifest Query**: Verifies artifacts are registered with `state=ok`
2. **File Existence**: Checks parquet files exist at manifest-registered paths
3. **Schema Validation**: Ensures required columns are present
4. **Row Count**: Verifies row count matches canonical grid size (509,209 for Mumbai)
5. **Grid Coverage**: Checks all canonical grid_ids are present
6. **Data Ranges**: Validates numeric values are within expected ranges
7. **Row Ordering**: Verifies rows match canonical grid order (deterministic)
8. **Null Values**: Reports any missing/null data

## Output

```
======================================================================
ARTIFACT CONSUMER TEST: landsat_signals
======================================================================
Description: Landsat-8 raw band values (B1-B11)
Expected schema: grid_id, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11
Grid size: 509,209 cells

Querying manifest for landsat_signals artifacts...
  ✓ Found 2 artifacts in manifest (state=ok)

[1/2] Validating 2018-01-01...
  Path: test_output/landsat8_full/2018-01-01.parquet
  ✓ VALID
    Rows: 509,209
    Columns: 12
    Warnings: 0

[2/2] Validating 2018-01-17...
  Path: test_output/landsat8_full/2018-01-17.parquet
  ✓ VALID
    Rows: 509,209
    Columns: 12
    Warnings: 0

======================================================================
VALIDATION SUMMARY
======================================================================
Artifact type:    landsat_signals
Total artifacts:  2
Valid:            2
Invalid:          0
Grid coverage:    509,209 cells expected

✓ All artifacts passed validation!
```

## Adding New Artifact Types

To add validation for new artifact types, update `ARTIFACT_SCHEMAS` in `test_artifact_consumer.py`:

```python
ARTIFACT_SCHEMAS = {
    "your_artifact_type": {
        "required_columns": ["grid_id", "col1", "col2"],
        "numeric_columns": ["col1", "col2"],
        "expected_range": (min_val, max_val),
        "description": "Brief description",
    },
}
```

Then add the type to CLI choices in `main()`.

## Exit Codes

- `0`: All artifacts valid
- `1`: One or more artifacts failed validation
