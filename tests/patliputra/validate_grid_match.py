"""Validate that new grid matches old grid coordinates."""
import pandas as pd
import numpy as np

print("=== NEW GRID (cache/static/grid_index.parquet) ===")
df_new = pd.read_parquet("cache/static/grid_index.parquet")
print(f"Total points: {len(df_new):,}")
print(f"Columns: {list(df_new.columns)}")

xs_new = np.sort(np.unique(df_new["x_m"].values))
ys_new = np.sort(np.unique(df_new["y_m"].values))
print(f"X spacing: {xs_new[1] - xs_new[0]:.6f} m")
print(f"Y spacing: {ys_new[1] - ys_new[0]:.6f} m")
print(f"X range: [{df_new['x_m'].min():.6f}, {df_new['x_m'].max():.6f}]")
print(f"Y range: [{df_new['y_m'].min():.6f}, {df_new['y_m'].max():.6f}]")
print(f"First 3 X values: {xs_new[:3]}")
print(f"First 3 Y values: {ys_new[:3]}")
print(f"\nrow_idx range: [{df_new['row_idx'].min()}, {df_new['row_idx'].max()}]")
print(f"row_idx null count: {df_new['row_idx'].isna().sum()}")

print("\n" + "="*60)
print("=== OLD GRID (from backup file) ===")
df_old = pd.read_parquet("tmp/landsat8_2018-03-06_signals.backup.parquet")
print(f"Total points: {len(df_old):,}")

xs_old = np.sort(np.unique(df_old["x"].values))
ys_old = np.sort(np.unique(df_old["y"].values))
print(f"X spacing: {xs_old[1] - xs_old[0]:.6f} m")
print(f"Y spacing: {ys_old[1] - ys_old[0]:.6f} m")
print(f"X range: [{df_old['x'].min():.6f}, {df_old['x'].max():.6f}]")
print(f"Y range: [{df_old['y'].min():.6f}, {df_old['y'].max():.6f}]")
print(f"First 3 X values: {xs_old[:3]}")
print(f"First 3 Y values: {ys_old[:3]}")

print("\n" + "="*60)
print("=== COORDINATE MATCH TEST ===")
matches = df_old.merge(df_new, left_on=["x", "y"], right_on=["x_m", "y_m"], how="inner")
print(f"Matching coordinates: {len(matches):,} / {len(df_old):,} ({100*len(matches)/len(df_old):.2f}%)")

if len(matches) > 0:
    print(f"\nSample matches (first 5):")
    for i in range(min(5, len(matches))):
        row = matches.iloc[i]
        print(f"  [{i+1}] Old (x,y): ({row['x']:.6f}, {row['y']:.6f}) -> row_idx: {row['row_idx']}")
else:
    print("\nNO MATCHES FOUND!")
    print("Checking if coordinates are close but not exact...")
    
    # Sample a few old points and find nearest new points
    sample_old = df_old.head(5)
    for i, row in sample_old.iterrows():
        old_x, old_y = row['x'], row['y']
        # Calculate distance to all new points
        distances = np.sqrt((df_new['x_m'] - old_x)**2 + (df_new['y_m'] - old_y)**2)
        min_dist_idx = distances.idxmin()
        min_dist = distances[min_dist_idx]
        nearest = df_new.loc[min_dist_idx]
        print(f"  Old ({old_x:.6f}, {old_y:.6f}) -> Nearest new ({nearest['x_m']:.6f}, {nearest['y_m']:.6f}), distance: {min_dist:.6f}m")

print("\n" + "="*60)
print("=== VALIDATION RESULT ===")
if len(matches) == len(df_old):
    print("✓ SUCCESS: All old coordinates match new grid exactly!")
    print(f"  Ready for simple join-based migration.")
elif len(matches) > 0:
    print(f"⚠ PARTIAL MATCH: {len(matches)}/{len(df_old)} coordinates match")
    print(f"  Missing: {len(df_old) - len(matches)} coordinates")
else:
    print("✗ FAILURE: No coordinate matches found!")
    print("  Grids have different origins. Need to regenerate grid or use spatial join.")
