# Running the LightGBM Trainer

This guide explains how to train the PM2.5 LightGBM model on your assembled data.

## Quick Start

```bash
cd /Users/shlokjain/pollu
source .venv/bin/activate

# Train on the full dataset (Jan 1-25, 2023)
python -m dwarka.trainer \
    --schema artifacts/v1/features/feature_schema.json \
    --train-dir artifacts/train \
    --val-dir artifacts/val \
    --output-dir dwarka/models/baseline_v1 \
    --model-version baseline_v1
```

## Data Setup

Your current split configuration:
- **Train**: January 1-25, 2023 (25 days, 3.7M rows)
- **Gap**: January 26-31, 2023 (prevents lag24 leakage)
- **Validation**: March 1-31, 2023 (31 days, 4.2M rows)

Files are in:
- Training data: `artifacts/train/*_training.parquet`
- Validation data: `artifacts/val/*_training.parquet`
- Schema: `artifacts/v1/features/feature_schema.json`

## Command Line Options

```
--schema PATH                  Path to frozen feature schema JSON
                              Default: artifacts/v1/features/feature_schema.json

--train-dir PATH              Directory with training parquet files
                              Default: artifacts/v1/training

--val-dir PATH                Directory with validation parquet files
                              Default: artifacts/v1/training

--output-dir PATH             Where to save model bundle
                              Default: dwarka/models/baseline_v1

--model-version STRING        Version label for metadata
                              Default: baseline_v1

--chunk-size DAYS             Training chunk size (default: 30 days)
                              Controls memory usage during training

--no-chunked                  Load all training data at once
                              NOT recommended for large datasets
                              Use only if you have 16+ GB RAM

--notes TEXT                  Optional notes to include in metadata
```

## Examples

### 1. Train on full data with chunks (default, memory-efficient)
```bash
python -m dwarka.trainer \
    --schema artifacts/v1/features/feature_schema.json \
    --train-dir artifacts/train \
    --val-dir artifacts/val \
    --output-dir dwarka/models/baseline_v1 \
    --model-version baseline_v1 \
    --chunk-size 30
```

### 2. Train on smaller chunks (fewer memory)
```bash
python -m dwarka.trainer \
    --schema artifacts/v1/features/feature_schema.json \
    --train-dir artifacts/train \
    --val-dir artifacts/val \
    --output-dir dwarka/models/baseline_v1 \
    --chunk-size 5
```

### 3. Train on full data in memory (if RAM available)
```bash
python -m dwarka.trainer \
    --schema artifacts/v1/features/feature_schema.json \
    --train-dir artifacts/train \
    --val-dir artifacts/val \
    --output-dir dwarka/models/baseline_v1 \
    --no-chunked
```

### 4. Train with custom metadata notes
```bash
python -m dwarka.trainer \
    --schema artifacts/v1/features/feature_schema.json \
    --train-dir artifacts/train \
    --val-dir artifacts/val \
    --output-dir dwarka/models/baseline_v1 \
    --model-version baseline_v1 \
    --notes "First baseline on 2023 Jan-Mar data, debug run"
```

## Output

The trainer will create a **model bundle** with 4 files:

```
dwarka/models/baseline_v1/
├── model.txt                    # LightGBM model (text format)
├── feature_schema.json          # Frozen feature list
├── training_metadata.json       # Training config and performance
└── manifest.json                # Model metadata and version info
```

## Monitor Training

During training, you'll see:
- Chunk loading progress
- LightGBM iteration metrics (every 20 iterations)
- Early stopping if validation loss plateaus
- Final model size and training time

## After Training

Once the model is trained:

1. **Evaluate on validation set**:
   ```bash
   python - <<'PY'
   import pickle
   from pathlib import Path
   from nalanda.feature_freeze import FrozenFeatureSchema
   from dwarka.evaluate import evaluate_model
   import lightgbm as lgb
   import pandas as pd
   
   # Load model
   model = lgb.Booster(model_file='dwarka/models/baseline_v1/model.txt')
   schema = FrozenFeatureSchema.load('dwarka/models/baseline_v1/feature_schema.json')
   
   # Load validation data
   val_files = sorted(Path('artifacts/val').glob('*_training.parquet'))
   val_df = pd.concat([pd.read_parquet(f) for f in val_files])
   
   # Get predictions
   X_val = schema.extract_X(val_df)
   y_pred = model.predict(X_val)
   y_true = val_df['pm25_t']
   
   # Evaluate
   results = evaluate_model(y_true, y_pred, 'baseline_v1')
   print(f"MAE: {results.overall.mae:.2f}")
   print(f"R²: {results.overall.r2:.3f}")
   PY
   ```

2. **Generate comparison report**:
   ```bash
   python - <<'PY'
   from dwarka.report import generate_full_report
   from pathlib import Path
   import pandas as pd
   import lightgbm as lgb
   
   # Load model and data
   model = lgb.Booster(model_file='dwarka/models/baseline_v1/model.txt')
   val_files = sorted(Path('artifacts/val').glob('*_training.parquet'))
   val_df = pd.concat([pd.read_parquet(f) for f in val_files])
   
   # Generate report (requires baselines, see dwarka/report.py)
   json_path, text_path = generate_full_report(val_df, baselines, model, {})
   print(f"Reports saved to {json_path} and {text_path}")
   PY
   ```

## Troubleshooting

**"Schema validation failed: Missing features"**
- The data pipeline hasn't computed all required features yet
- Check `artifacts/v1/features/feature_list.txt` vs actual data columns
- You may need to run data assembly/enrichment steps

**"Out of memory"**
- Use smaller `--chunk-size` (e.g., 5 days instead of 30)
- Or run with `--no-chunked` on a machine with more RAM

**"No training data found"**
- Check that `artifacts/train/` contains `*_training.parquet` files
- Run `ls -la artifacts/train/ | head` to verify

**"Model not improving"**
- Check feature distributions are reasonable
- May need feature engineering or hyperparameter tuning
- See `dwarka/config.py` for LightGBM parameters
