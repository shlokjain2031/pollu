# Sky View Factor (SVF) Implementation Notes

## What is SVF?

Sky View Factor quantifies the **fraction of the sky hemisphere visible from a point**. It ranges from 0 (completely obstructed) to 1 (open sky in all directions).

### Formula

$$\text{SVF} = \frac{1}{n} \sum_{i=1}^{n} \cos^2(\alpha_i)$$

where:
- $n$ = number of azimuth directions sampled
- $\alpha_i$ = maximum horizon angle in direction $i$ (measured from horizontal plane)

### Physical Meaning

- **SVF = 1.0**: Open field (no obstructions)
- **SVF = 0.5**: Half the sky is blocked (e.g., street canyon)
- **SVF = 0.0**: Completely enclosed (e.g., inside building)

## Implementation Approaches

### 1. Horizon-Based Algorithm (Proper Method)

**Method**: Ray-cast in multiple azimuth directions, find maximum elevation angle, integrate.

**Pros**:
- Physically accurate
- Produces meaningful 0-1 values
- Respects urban geometry (buildings, topography)

**Cons**:
- Computationally expensive (O(n_directions × search_radius))
- Earth Engine memory limits (~50MB download, computation timeouts)
- Requires careful parameter tuning

**Implementation**: `patliputra/svf_earth_engine.py`

### 2. Topographic Position Index (TPI) Proxy

**Method**: Compute mean elevation in neighborhood, subtract from center elevation.

```python
tpi = dsm.subtract(
    dsm.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius=50, units='meters')
    )
)
```

**Pros**:
- Fast (single neighborhood operation)
- Works with Earth Engine limits

**Cons**:
- Not SVF (arbitrary units)
- Sensitive to scale choice
- Doesn't represent sky visibility

### 3. Local Standard Deviation Proxy

**Method**: Compute elevation variability in neighborhood.

```python
roughness = dsm.reduceNeighborhood(
    reducer=ee.Reducer.stdDev(),
    kernel=ee.Kernel.circle(radius=50, units='meters')
)
```

**Pros**:
- Very fast
- Captures urban roughness

**Cons**:
- Not physically meaningful
- High values = rough terrain (inverse relationship to SVF!)
- Can't distinguish between hill and valley

## Recommended Workflow

### For Research/Analysis
Use **horizon-based SVF** (`svf_earth_engine.py`) with:
- 16 directions (22.5° azimuth spacing)
- 100m search radius
- 30m resolution output

### For Rapid Prototyping
Use **TPI proxy** as SVF surrogate:
- Normalize TPI to 0-1 range
- Interpret high TPI = low SVF (elevated = more exposed)
- Test correlations with PM2.5 before committing to full SVF

## Computational Considerations

### Earth Engine Limits
- **Download limit**: 50MB per request
  - Solution: Tile the area (like Landsat processing)
- **Computation timeout**: ~5 minutes
  - Solution: Reduce search radius or directions
  - Use `ee.batch.Export` for large areas

### Parameters Trade-offs

| Parameter | More = Better Quality | More = Slower |
|-----------|----------------------|---------------|
| `n_directions` | ✓ (smoother integration) | ✓ (linear scaling) |
| `search_radius_m` | ✓ (captures distant obstacles) | ✓ (quadratic scaling) |
| `resolution_m` | Lower = more detail | Lower = exponential cost |

**Recommended for Mumbai (22km × 42km)**:
- Resolution: 30m (matches Landsat)
- Directions: 8-16 (good balance)
- Search radius: 50-100m (captures street-scale obstructions)

## Alternative: Pre-computed SVF Datasets

If Earth Engine proves too slow, consider:

1. **Urban Atlas**: Some cities have pre-computed SVF layers
2. **Local computation**: Download DSM once, compute SVF locally with Python
3. **QGIS UMEP**: Urban Multi-scale Environmental Predictor plugin

## Usage Example

```bash
# Compute SVF for Mumbai 2023
python patliputra/svf_earth_engine.py \
  --year 2023 \
  --output cache/mumbai_svf_2023.tif \
  --directions 16 \
  --radius 100

# Then sample at grid points (like Landsat)
python scripts/sample_svf_to_grid.py \
  --svf-raster cache/mumbai_svf_2023.tif \
  --grid-parquet data/mumbai/grid_30m.parquet \
  --output cache/mumbai_svf_grid.parquet
```

## References

1. Zakšek, K., Oštir, K., & Kokalj, Ž. (2011). "Sky-View Factor as a Relief Visualization Technique". *Remote Sensing*, 3(2), 398-415.

2. Yokoyama, R., Shirasawa, M., & Pike, R. J. (2002). "Visualizing Topography by Openness: A New Application of Image Processing to Digital Elevation Models". *Photogrammetric Engineering & Remote Sensing*, 68(3), 257-265.

3. Helbig, N., Löwe, H., & Lehning, M. (2009). "Radiosity Approach for the Shortwave Surface Radiation Balance in Complex Terrain". *Journal of the Atmospheric Sciences*, 66(9), 2900-2912.
