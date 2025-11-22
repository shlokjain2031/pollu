# SVF Interpretation for Urban Air Quality

## Sky View Factor (SVF) Values and Their Meaning

### SVF Physical Interpretation

| SVF Range | Environment | H/W Ratio | Air Quality Impact |
|-----------|-------------|-----------|-------------------|
| 0.9 - 1.0 | Open space (parks, fields) | < 0.1 | Excellent ventilation, rapid dispersion |
| 0.7 - 0.9 | Low-rise residential | 0.1 - 0.5 | Good ventilation |
| 0.5 - 0.7 | Urban residential | 0.5 - 1.0 | Moderate ventilation |
| 0.3 - 0.5 | **Street canyon** | 1.0 - 2.0 | **Poor ventilation, pollutant trapping** |
| 0.1 - 0.3 | Deep canyon | 2.0 - 3.0 | Very poor ventilation |
| < 0.1 | Tunnel/covered | > 3.0 | Minimal ventilation |

### Street Canyon Detection by Pixel Size

**For Mumbai pollution analysis:**

#### 0.5m Pixels
- **Pros**: Captures every lane, exact building edges
- **Cons**: 
  - Too detailed for PM2.5 (which disperses ~100m scale)
  - File too large (40GB+)
  - Noise from small gaps between buildings
- **SVF < 0.4**: Individual pixel is in narrow lane
- **Use case**: Pedestrian-level exposure, hyperlocal studies

#### 1m Pixels
- **Pros**: Good detail, captures most streets
- **Cons**: Still very large file (~10GB), overkill for PM2.5
- **SVF < 0.4**: Consistent street canyon
- **Use case**: Street-level air quality mapping

#### 2m Pixels ⭐ **RECOMMENDED FOR STREETS**
- **Pros**: 
  - Captures local streets (6m+) accurately
  - Pixel size matches typical street width/2
  - More manageable file size (~2.5GB)
  - Good signal-to-noise for pollution modeling
- **Cons**: Misses very narrow lanes (< 4m)
- **SVF < 0.5**: Definite street canyon
- **SVF 0.5-0.6**: Street with tall buildings
- **SVF > 0.7**: Open or wide street
- **Use case**: Urban PM2.5 modeling, traffic pollution studies

#### 5m Pixels ⭐ **RECOMMENDED FOR PM2.5 ANALYSIS**
- **Pros**:
  - Matches PM2.5 sensor spacing (typically 100m+ apart)
  - Captures neighborhood-scale urban geometry
  - Reasonable file size (~350MB)
  - Aggregates natural variability
  - Won't fail Earth Engine limits
- **Cons**: Misses individual narrow streets
- **SVF < 0.5**: Dense urban area with street canyons
- **SVF 0.5-0.7**: Mixed urban (some open space)
- **SVF > 0.7**: Low-rise or open area
- **Use case**: **City-scale PM2.5 modeling (YOUR USE CASE)**

#### 10m Pixels (Alternative)
- **Pros**: Very fast, small file (~90MB)
- **Cons**: Too coarse for street-level features
- **SVF < 0.6**: Urban canyon neighborhood
- **Use case**: Regional studies, preliminary analysis

## Recommendation for Your PM2.5 Study

### Use **5m resolution** because:

1. **Scale Matching**: PM2.5 sensors are spaced 100m-1km apart, SVF at 5m averages over 20×20m area = appropriate for sensor footprint

2. **Computational Feasibility**: ~350MB file, should work with Earth Engine direct download (no tiling needed)

3. **Physical Relevance**: 
   - 5m pixel captures "block-scale" urban geometry
   - PM2.5 disperses over 50-200m scale → 5m SVF represents local ventilation potential
   - Aggregates away noise from individual building edges

4. **Interpretability**:
   - **SVF < 0.5**: Dense urban canyon block → expect 20-30% higher PM2.5
   - **SVF 0.5-0.7**: Typical urban → baseline PM2.5
   - **SVF > 0.7**: Open/green area → 10-20% lower PM2.5

### Analysis Workflow with 5m SVF

```python
# After downloading 5m SVF raster:

# 1. Sample at PM2.5 sensor locations
sensors_gdf['svf'] = sample_raster_at_points(svf_raster, sensors_gdf)

# 2. Classify urban morphology
sensors_gdf['morphology'] = pd.cut(
    sensors_gdf['svf'],
    bins=[0, 0.4, 0.6, 0.8, 1.0],
    labels=['deep_canyon', 'street_canyon', 'urban_open', 'suburban']
)

# 3. Analyze PM2.5 by morphology
pm25_by_morphology = sensors_gdf.groupby('morphology')['pm25'].describe()

# 4. Control for SVF in regression
# PM2.5 ~ traffic + landsat_ndvi + temperature + SVF + ...
```

## Expected SVF-PM2.5 Relationship

From literature (Ng & Ren 2015, Yuan et al. 2017):
- **1.0 → 0.5 SVF** (open → canyon): PM2.5 increases 15-25%
- **Mechanism**: Reduced ventilation → slower dispersion → accumulation

### Mumbai-Specific Factors

**High SVF areas** (open, good ventilation):
- Parks: Shivaji Park, Oval Maidan
- Coastal: Marine Drive, Worli Sea Face
- Expected: Lower PM2.5 (except near major roads)

**Low SVF areas** (canyons, poor ventilation):
- Dharavi (dense slums, narrow lanes)
- Fort/CST (old city, narrow streets)
- Dadar (dense commercial)
- Expected: Higher PM2.5 if also high traffic

## References

1. Ng, E., & Ren, C. (2015). "The Urban Climatic Map: A Methodology for Sustainable Urban Planning". Routledge.

2. Yuan, C., Ng, E., & Norford, L. K. (2014). "Improving air quality in high-density cities by understanding the relationship between air pollutant dispersion and urban morphologies". *Building and Environment*, 71, 245-258.

3. Hang, J., et al. (2015). "The influence of building height variability on pollutant dispersion and pedestrian ventilation in idealized high-rise urban areas". *Building and Environment*, 56, 346-360.
