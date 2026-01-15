"""Comprehensive data quality check for PM2.5 training data.

Diagnoses issues that could be causing poor model performance (MAE > 20).
Target: MAE < 10 requires high-quality, consistent data.

Usage:
    python scripts/data_quality_check.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_sample_files(split: str, n_files: int = 10) -> pd.DataFrame:
    """Load a sample of files from a split."""
    split_dir = PROJECT_ROOT / "artifacts" / split
    files = sorted(list(split_dir.glob("*_training.parquet")))[:n_files]
    
    if not files:
        raise ValueError(f"No files found in {split_dir}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def check_target_quality(df: pd.DataFrame) -> Dict:
    """Check target variable (pm25_t) quality."""
    target = df['pm25_t']
    
    results = {
        'count': len(target),
        'nan_count': target.isna().sum(),
        'nan_pct': (target.isna().sum() / len(target)) * 100,
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'median': target.median(),
        'q25': target.quantile(0.25),
        'q75': target.quantile(0.75),
        'negative_count': (target < 0).sum(),
        'zero_count': (target == 0).sum(),
        'extreme_high': (target > 500).sum(),
        'very_high': (target > 200).sum(),
        'duplicates': target.duplicated().sum(),
    }
    
    return results


def check_feature_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Check NaN rates for all features."""
    exclude = ['timestamp', 'row_idx', 'pm25_t', 'target_trust', 'grid_id', 'n_sensors', 
               'landsat_date', 'sentinel_date']
    features = [c for c in df.columns if c not in exclude]
    
    stats = []
    for feat in features:
        nan_count = df[feat].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        dtype = str(df[feat].dtype)
        stats.append({
            'feature': feat,
            'nan_count': nan_count,
            'nan_pct': nan_pct,
            'dtype': dtype,
        })
    
    return pd.DataFrame(stats).sort_values('nan_pct', ascending=False)


def check_lag_feature_validity(df: pd.DataFrame) -> Dict:
    """Check if lag features are valid (not just all NaN)."""
    lag_features = [c for c in df.columns if 'lag' in c.lower() or 'minus' in c.lower()]
    
    results = {}
    for feat in lag_features:
        if feat in df.columns:
            valid_pct = (df[feat].notna().sum() / len(df)) * 100
            results[feat] = valid_pct
    
    return results


def check_modis_coverage(df: pd.DataFrame) -> Dict:
    """Check MODIS feature coverage."""
    modis_features = ['aod_047', 'aod_055', 'aod_uncertainty', 'column_wv']
    presence_flags = ['has_aod_047', 'has_aod_055']
    
    results = {}
    
    for feat in modis_features:
        if feat in df.columns:
            valid = df[feat].notna().sum()
            results[f'{feat}_valid'] = valid
            results[f'{feat}_pct'] = (valid / len(df)) * 100
    
    for flag in presence_flags:
        if flag in df.columns:
            present = df[flag].sum()
            results[f'{flag}_present'] = present
            results[f'{flag}_pct'] = (present / len(df)) * 100
    
    return results


def check_temporal_coverage(split: str) -> Dict:
    """Check temporal coverage and gaps."""
    split_dir = PROJECT_ROOT / "artifacts" / split
    files = sorted(list(split_dir.glob("*_training.parquet")))
    
    if not files:
        return {}
    
    dates = [f.stem.split('_')[0] for f in files]
    dates_dt = pd.to_datetime(dates)
    
    return {
        'n_files': len(files),
        'first_date': dates[0],
        'last_date': dates[-1],
        'date_range_days': (dates_dt[-1] - dates_dt[0]).days,
        'mean_file_size_mb': np.mean([f.stat().st_size / 1024 / 1024 for f in files[:100]]),
    }


def check_split_contamination() -> Dict:
    """Check if train/val/test splits overlap."""
    splits = ['train', 'val', 'test']
    date_sets = {}
    
    for split in splits:
        split_dir = PROJECT_ROOT / "artifacts" / split
        files = list(split_dir.glob("*_training.parquet"))
        dates = set(f.stem.split('_')[0] for f in files)
        date_sets[split] = dates
    
    results = {
        'train_count': len(date_sets.get('train', set())),
        'val_count': len(date_sets.get('val', set())),
        'test_count': len(date_sets.get('test', set())),
    }
    
    # Check overlaps
    if 'train' in date_sets and 'val' in date_sets:
        overlap_train_val = date_sets['train'] & date_sets['val']
        results['train_val_overlap'] = len(overlap_train_val)
        results['train_val_overlap_dates'] = sorted(list(overlap_train_val))[:10]
    
    if 'train' in date_sets and 'test' in date_sets:
        overlap_train_test = date_sets['train'] & date_sets['test']
        results['train_test_overlap'] = len(overlap_train_test)
    
    if 'val' in date_sets and 'test' in date_sets:
        overlap_val_test = date_sets['val'] & date_sets['test']
        results['val_test_overlap'] = len(overlap_val_test)
    
    return results


def check_target_correlation(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Check feature correlation with target."""
    exclude = ['timestamp', 'row_idx', 'pm25_t', 'target_trust', 'grid_id', 'n_sensors',
               'landsat_date', 'sentinel_date']
    features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    
    correlations = []
    for feat in features:
        try:
            corr = df[[feat, 'pm25_t']].corr().iloc[0, 1]
            correlations.append({
                'feature': feat,
                'correlation': corr,
                'abs_correlation': abs(corr),
            })
        except:
            pass
    
    return pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False).head(top_n)


def check_data_drift(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    """Check if train and validation distributions differ."""
    results = {}
    
    # Target distribution
    results['target_mean_train'] = train_df['pm25_t'].mean()
    results['target_mean_val'] = val_df['pm25_t'].mean()
    results['target_std_train'] = train_df['pm25_t'].std()
    results['target_std_val'] = val_df['pm25_t'].std()
    results['target_drift_pct'] = abs(results['target_mean_train'] - results['target_mean_val']) / results['target_mean_train'] * 100
    
    return results


def main():
    """Run comprehensive data quality checks."""
    print("=" * 80)
    print("DATA QUALITY CHECK - PM2.5 TRAINING DATA")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    # 1. Check split contamination
    print("1. CHECKING SPLIT CONTAMINATION")
    print("-" * 80)
    split_info = check_split_contamination()
    print(f"Train files: {split_info['train_count']}")
    print(f"Val files: {split_info['val_count']}")
    print(f"Test files: {split_info['test_count']}")
    print(f"\nTrain-Val overlap: {split_info.get('train_val_overlap', 0)} dates")
    if split_info.get('train_val_overlap', 0) > 0:
        print(f"⚠️  WARNING: Train and Val sets overlap!")
        print(f"   Example dates: {split_info.get('train_val_overlap_dates', [])}")
    print(f"Train-Test overlap: {split_info.get('train_test_overlap', 0)} dates")
    print(f"Val-Test overlap: {split_info.get('val_test_overlap', 0)} dates")
    print()
    
    # 2. Load sample data
    print("2. LOADING SAMPLE DATA (10 files per split)")
    print("-" * 80)
    train_sample = load_sample_files('train', n_files=10)
    val_sample = load_sample_files('val', n_files=10)
    print(f"Train sample: {len(train_sample):,} rows, {len(train_sample.columns)} columns")
    print(f"Val sample: {len(val_sample):,} rows, {len(val_sample.columns)} columns")
    print()
    
    # 3. Check target quality
    print("3. TARGET VARIABLE QUALITY (pm25_t)")
    print("-" * 80)
    target_stats = check_target_quality(train_sample)
    print(f"Count: {target_stats['count']:,}")
    print(f"NaN: {target_stats['nan_count']:,} ({target_stats['nan_pct']:.2f}%)")
    print(f"Mean: {target_stats['mean']:.2f} μg/m³")
    print(f"Std: {target_stats['std']:.2f}")
    print(f"Min: {target_stats['min']:.2f}, Max: {target_stats['max']:.2f}")
    print(f"Median: {target_stats['median']:.2f}")
    print(f"Q25: {target_stats['q25']:.2f}, Q75: {target_stats['q75']:.2f}")
    print(f"\nData quality issues:")
    print(f"  Negative values: {target_stats['negative_count']:,}")
    print(f"  Zero values: {target_stats['zero_count']:,}")
    print(f"  > 200 μg/m³: {target_stats['very_high']:,}")
    print(f"  > 500 μg/m³: {target_stats['extreme_high']:,}")
    print(f"  Duplicates: {target_stats['duplicates']:,}")
    
    if target_stats['negative_count'] > 0:
        print("  ⚠️  WARNING: Negative PM2.5 values found!")
    if target_stats['extreme_high'] > 0:
        print("  ⚠️  WARNING: Extreme PM2.5 values (>500) found!")
    print()
    
    # 4. Feature completeness
    print("4. FEATURE COMPLETENESS (Top 15 with most NaNs)")
    print("-" * 80)
    feature_stats = check_feature_completeness(train_sample)
    print(feature_stats.head(15).to_string(index=False))
    
    critical_missing = feature_stats[feature_stats['nan_pct'] > 50]
    if len(critical_missing) > 0:
        print(f"\n⚠️  WARNING: {len(critical_missing)} features have >50% NaN!")
        print(f"   Features: {critical_missing['feature'].tolist()}")
    print()
    
    # 5. MODIS coverage
    print("5. MODIS FEATURE COVERAGE")
    print("-" * 80)
    modis_stats = check_modis_coverage(train_sample)
    for key, value in modis_stats.items():
        if 'pct' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:,}")
    
    aod_coverage = modis_stats.get('aod_055_pct', 0)
    if aod_coverage < 10:
        print(f"\n⚠️  WARNING: MODIS AOD coverage is only {aod_coverage:.1f}%")
        print("   This might not provide enough signal for improvement")
    print()
    
    # 6. Lag feature validity
    print("6. LAG FEATURE VALIDITY (Top 10)")
    print("-" * 80)
    lag_stats = check_lag_feature_validity(train_sample)
    sorted_lags = sorted(lag_stats.items(), key=lambda x: x[1])[:10]
    for feat, valid_pct in sorted_lags:
        print(f"{feat}: {valid_pct:.2f}% valid")
    
    critical_lags = {k: v for k, v in lag_stats.items() if v < 50}
    if critical_lags:
        print(f"\n⚠️  WARNING: {len(critical_lags)} lag features have <50% valid data")
    print()
    
    # 7. Target correlation
    print("7. TOP 20 FEATURES BY CORRELATION WITH TARGET")
    print("-" * 80)
    correlations = check_target_correlation(train_sample, top_n=20)
    print(correlations.to_string(index=False))
    
    weak_top_corr = correlations.iloc[0]['abs_correlation'] if len(correlations) > 0 else 0
    if weak_top_corr < 0.3:
        print(f"\n⚠️  WARNING: Strongest correlation is only {weak_top_corr:.3f}")
        print("   Features may not be predictive enough")
    print()
    
    # 8. Data drift
    print("8. TRAIN-VAL DISTRIBUTION DRIFT")
    print("-" * 80)
    drift_stats = check_data_drift(train_sample, val_sample)
    print(f"Target mean (train): {drift_stats['target_mean_train']:.2f}")
    print(f"Target mean (val): {drift_stats['target_mean_val']:.2f}")
    print(f"Target std (train): {drift_stats['target_std_train']:.2f}")
    print(f"Target std (val): {drift_stats['target_std_val']:.2f}")
    print(f"Distribution drift: {drift_stats['target_drift_pct']:.2f}%")
    
    if drift_stats['target_drift_pct'] > 10:
        print(f"\n⚠️  WARNING: Train-Val target distribution differs by {drift_stats['target_drift_pct']:.1f}%")
        print("   This could explain poor validation performance")
    print()
    
    # 9. Temporal coverage
    print("9. TEMPORAL COVERAGE")
    print("-" * 80)
    for split in ['train', 'val', 'test']:
        temporal = check_temporal_coverage(split)
        if temporal:
            print(f"{split.upper()}:")
            print(f"  Files: {temporal['n_files']}")
            print(f"  Date range: {temporal['first_date']} to {temporal['last_date']}")
            print(f"  Days covered: {temporal['date_range_days']}")
            print(f"  Avg file size: {temporal['mean_file_size_mb']:.2f} MB")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    issues = []
    
    if split_info.get('train_val_overlap', 0) > 0:
        issues.append("❌ Train-Val contamination detected")
    
    if target_stats['negative_count'] > 0 or target_stats['extreme_high'] > 0:
        issues.append("❌ Target variable has outliers/invalid values")
    
    if len(critical_missing) > 0:
        issues.append(f"⚠️  {len(critical_missing)} features have >50% missing data")
    
    if aod_coverage < 10:
        issues.append(f"⚠️  MODIS coverage is low ({aod_coverage:.1f}%)")
    
    if drift_stats['target_drift_pct'] > 10:
        issues.append(f"⚠️  Train-Val distribution drift ({drift_stats['target_drift_pct']:.1f}%)")
    
    if weak_top_corr < 0.3:
        issues.append("⚠️  Weak feature-target correlations")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No critical issues found!")
    
    print("\nTo achieve MAE < 10:")
    print("1. Fix data contamination if present")
    print("2. Remove/cap outliers in target variable")
    print("3. Consider adding spatial features (grid_id as categorical)")
    print("4. Verify lag features are correctly computed")
    print("5. Check if temporal split is appropriate (avoid train-test leakage)")
    print()


if __name__ == "__main__":
    main()
