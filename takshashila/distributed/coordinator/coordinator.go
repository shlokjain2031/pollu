package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

const maxRequestBodyBytes = 8 << 10

// Coordinator serves /predict by layering: in-memory result cache → Redis →
// request deduplication → on-demand inference via kashi.
type Coordinator struct {
	cfg       *Config
	cache     *ResultCache
	dedup     *Deduplicator
	redis     *redis.Client // nil disables the Redis tier (tests)
	inference *InferenceClient

	metaMu         sync.RWMutex
	modelVersion   string
	featureVersion string
	pointCount     int // 0 = unknown (kashi not yet reachable)

	redisUp     bool
	inferenceUp bool
	statusMu    sync.RWMutex
}

// NewCoordinator wires the coordinator. Redis and kashi may be down at
// startup; the coordinator degrades and recovers via watchDependencies.
func NewCoordinator(cfg *Config, redisClient *redis.Client, inference *InferenceClient) *Coordinator {
	c := &Coordinator{
		cfg:            cfg,
		cache:          NewResultCache(cfg.ResultCacheTTL),
		dedup:          NewDeduplicator(),
		redis:          redisClient,
		inference:      inference,
		modelVersion:   "unknown",
		featureVersion: "unknown",
	}
	c.refreshMeta(context.Background())
	return c
}

// refreshMeta pulls model/coverage metadata from kashi. Safe to call repeatedly.
func (c *Coordinator) refreshMeta(ctx context.Context) {
	metaCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	meta, err := c.inference.Meta(metaCtx)
	if err != nil {
		return
	}
	c.metaMu.Lock()
	c.modelVersion = meta.ModelVersion
	c.featureVersion = meta.FeatureVersion
	c.pointCount = meta.PointCount
	c.metaMu.Unlock()
}

// versions returns the current model/feature versions and feature-point count.
func (c *Coordinator) versions() (string, string, int) {
	c.metaMu.RLock()
	defer c.metaMu.RUnlock()
	return c.modelVersion, c.featureVersion, c.pointCount
}

// watchDependencies probes Redis and kashi until ctx is cancelled, keeping
// /health accurate and re-fetching metadata after kashi restarts. It also
// sweeps expired result-cache entries (Get only treats them as misses;
// without the sweep the map would grow unboundedly under distinct-key load).
func (c *Coordinator) watchDependencies(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	lastSweep := time.Now()
	for {
		if time.Since(lastSweep) >= time.Minute {
			if evicted := c.cache.EvictExpired(); evicted > 0 {
				log.Printf("result cache: evicted %d expired entries", evicted)
			}
			lastSweep = time.Now()
		}
		probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		redisUp := false
		if c.redis != nil {
			redisUp = c.redis.Ping(probeCtx).Err() == nil
		}
		inferenceUp := c.inference.Healthy(probeCtx)
		cancel()

		c.statusMu.Lock()
		c.redisUp = redisUp
		c.inferenceUp = inferenceUp
		c.statusMu.Unlock()

		if _, _, pointCount := c.versions(); inferenceUp && pointCount == 0 {
			c.refreshMeta(ctx)
		}

		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// PredictRequest is a lat/lon query for a given hour.
type PredictRequest struct {
	Lat           *float64 `json:"lat"`
	Lon           *float64 `json:"lon"`
	TimestampHour string   `json:"timestamp_hour"`
}

// PredictResponse is the success payload.
type PredictResponse struct {
	RequestID string  `json:"request_id"`
	Lat       float64 `json:"lat"`
	Lon       float64 `json:"lon"`
	PM25      float64 `json:"pm25"`
	Source    string  `json:"source"` // cache | redis | dedup | inference
}

// canonicalLocation rounds a coordinate to ~11 m and formats it as the stable
// "lat,lon" string used in all cache/dedup keys. Rounding lets nearby repeat
// clicks share a cache entry while staying far finer than the 250 m feature grid.
func canonicalLocation(lat, lon float64) string {
	round := func(v float64) float64 { return math.Round(v*1e4) / 1e4 }
	return fmt.Sprintf("%.4f,%.4f", round(lat), round(lon))
}

// canonicalHour parses a timestamp, truncates to the hour, and returns the
// canonical UTC form used in all cache keys ("2006-01-02T15:04:05Z").
func canonicalHour(raw string) (string, PredictionError) {
	var t time.Time
	var err error
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04:05"} {
		if t, err = time.Parse(layout, raw); err == nil {
			break
		}
	}
	if err != nil {
		return "", NewInvalidRequestError(fmt.Sprintf("invalid timestamp_hour %q (want RFC3339, e.g. 2026-06-09T14:00:00Z)", raw))
	}
	t = t.UTC().Truncate(time.Hour)
	if t.Year() < 2015 || t.Year() > 2035 {
		return "", NewInvalidRequestError(fmt.Sprintf("timestamp_hour %q outside supported window 2015-2035", raw))
	}
	return t.Format("2006-01-02T15:04:05Z"), nil
}

func redisKey(location, timestampHour string) string {
	return "prediction:" + location + ":" + timestampHour
}

// handlePredict is the request flow: validate → cache → redis → dedup → kashi.
func (c *Coordinator) handlePredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	requestID := uuid.New().String()

	if r.Method != http.MethodPost {
		writeError(w, requestID, NewInvalidRequestError("method not allowed, use POST"), http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(w, requestID, NewInvalidRequestError("request body too large"), http.StatusRequestEntityTooLarge)
			return
		}
		writeError(w, requestID, NewInvalidRequestError(fmt.Sprintf("invalid JSON: %v", err)), 0)
		return
	}

	pm25, lat, lon, source, predErr := c.predict(r.Context(), requestID, &req)

	elapsed := time.Since(start)
	model, feature, _ := c.versions()
	if globalMetrics != nil {
		status := "success"
		if predErr != nil {
			status = "error"
			globalMetrics.ErrorsTotal.WithLabelValues(predErr.Type().String(), model, feature).Inc()
		}
		globalMetrics.RequestsTotal.WithLabelValues(source, status, model, feature).Inc()
		globalMetrics.RequestLatencySeconds.Observe(elapsed.Seconds())
	}

	if predErr != nil {
		log.Printf("[%s] %s ts=%s err=%s (%dms)", requestID, predErr.Type(), req.TimestampHour, predErr.Error(), elapsed.Milliseconds())
		writeError(w, requestID, predErr, 0)
		return
	}

	if source != "cache" { // cache hits are counted in metrics, not logged
		log.Printf("[%s] OK lat=%.4f lon=%.4f pm25=%.2f source=%s (%dms)", requestID, lat, lon, pm25, source, elapsed.Milliseconds())
	}
	writeJSON(w, http.StatusOK, PredictResponse{RequestID: requestID, Lat: lat, Lon: lon, PM25: pm25, Source: source})
}

// predict validates the request, then walks the cache tiers.
// Returns (pm25, lat, lon, source, error).
func (c *Coordinator) predict(ctx context.Context, requestID string, req *PredictRequest) (float64, float64, float64, string, PredictionError) {
	ctx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()

	tsHour, predErr := canonicalHour(req.TimestampHour)
	if predErr != nil {
		return 0, 0, 0, "validation", predErr
	}

	if req.Lat == nil || req.Lon == nil {
		return 0, 0, 0, "validation", NewInvalidRequestError("provide both lat and lon")
	}
	lat, lon := *req.Lat, *req.Lon
	if math.IsNaN(lat) || math.IsNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180 {
		return 0, lat, lon, "validation", NewInvalidRequestError(fmt.Sprintf("lat/lon out of range: (%v, %v)", lat, lon))
	}

	model, feature, _ := c.versions()
	location := canonicalLocation(lat, lon)
	key := DedupKey{Location: location, TimestampHour: tsHour, ModelVersion: model, FeatureVersion: feature}
	cacheKey := CacheKey(key)

	// Tier 1: in-memory result cache (also caches recent errors).
	if pm25, cachedErr, hit := c.cache.Get(cacheKey); hit {
		return pm25, lat, lon, "cache", cachedErr
	}

	// Tier 2: Redis.
	if pm25, ok := c.redisGet(ctx, location, tsHour); ok {
		c.cache.Set(cacheKey, pm25)
		return pm25, lat, lon, "redis", nil
	}

	// Tier 3: deduplicated on-demand inference.
	entry, isLeader := c.dedup.AcquireLeader(ctx, key, requestID)
	if !isLeader {
		if globalMetrics != nil {
			globalMetrics.DedupedRequestsTotal.Inc()
		}
		result, err := c.dedup.WaitForResult(ctx, entry)
		if err != nil {
			return 0, lat, lon, "dedup", ClassifyError(err)
		}
		return result.PM25, lat, lon, "dedup", ClassifyError(result.Error)
	}

	pm25, source, predErr := c.computeAsLeader(ctx, key, cacheKey, lat, lon)
	return pm25, lat, lon, source, predErr
}

// computeAsLeader calls kashi and broadcasts the result to dedup followers.
// The dedup entry is always completed, even on panic, so followers never hang.
func (c *Coordinator) computeAsLeader(ctx context.Context, key DedupKey, cacheKey CacheKey, lat, lon float64) (pm25 float64, source string, predErr PredictionError) {
	start := time.Now()
	completed := false
	defer func() {
		if !completed {
			c.dedup.Complete(key, &DedupResult{Error: NewUnknownError("leader aborted")})
		}
	}()

	value, err := c.inference.Predict(ctx, lat, lon, key.TimestampHour)
	predErr = ClassifyError(err)

	if globalMetrics != nil {
		globalMetrics.LeaderComputeLatencySeconds.Observe(time.Since(start).Seconds())
	}

	if predErr == nil {
		c.cache.Set(cacheKey, value)
		c.redisSet(key.Location, key.TimestampHour, value)
	} else if predErr.Cacheable() {
		c.cache.SetError(cacheKey, predErr)
	}

	c.dedup.Complete(key, &DedupResult{PM25: value, Error: predErr})
	completed = true
	return value, "inference", predErr
}

// redisGet reads a prediction from Redis; any failure is a cache miss.
func (c *Coordinator) redisGet(ctx context.Context, location, tsHour string) (float64, bool) {
	if c.redis == nil {
		return 0, false
	}
	getCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	raw, err := c.redis.Get(getCtx, redisKey(location, tsHour)).Result()
	if err != nil {
		if err != redis.Nil {
			log.Printf("redis get failed: %v", err)
		}
		return 0, false
	}
	pm25, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		log.Printf("redis value corrupt for %s: %q", redisKey(location, tsHour), raw)
		return 0, false
	}
	return pm25, true
}

// redisSet stores a prediction asynchronously; failures only log.
func (c *Coordinator) redisSet(location, tsHour string, pm25 float64) {
	if c.redis == nil {
		return
	}
	go func() {
		setCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := c.redis.Set(setCtx, redisKey(location, tsHour), strconv.FormatFloat(pm25, 'f', 2, 64), c.cfg.RedisTTL).Err(); err != nil {
			log.Printf("redis set failed: %v", err)
		}
	}()
}

// handleHealth reports coordinator and dependency status.
func (c *Coordinator) handleHealth(w http.ResponseWriter, r *http.Request) {
	c.statusMu.RLock()
	redisUp, inferenceUp := c.redisUp, c.inferenceUp
	c.statusMu.RUnlock()
	model, feature, pointCount := c.versions()

	status := "healthy"
	if !inferenceUp || !redisUp {
		status = "degraded"
	}

	dedupStats := c.dedup.Stats()
	cacheStats := c.cache.Stats()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":          status,
		"redis_up":        redisUp,
		"inference_up":    inferenceUp,
		"model_version":   model,
		"feature_version": feature,
		"point_count":     pointCount,
		"dedup": map[string]int{
			"active_computations": dedupStats.ActiveComputations,
			"total_followers":     dedupStats.TotalFollowers,
		},
		"cache": map[string]int{
			"total_entries":   cacheStats.TotalEntries,
			"expired_entries": cacheStats.ExpiredEntries,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(payload)
}

// writeError sends the standard error envelope. statusOverride of 0 uses the
// error's own HTTP status.
func writeError(w http.ResponseWriter, requestID string, predErr PredictionError, statusOverride int) {
	status := predErr.HTTPStatus()
	if statusOverride != 0 {
		status = statusOverride
	}
	if predErr.Retryable() && predErr.Cacheable() {
		w.Header().Set("Retry-After", strconv.Itoa(int(predErr.CacheTTL().Seconds())))
	}
	body := NewErrorResponse(predErr)
	body.HTTPStatus = status
	writeJSON(w, status, map[string]any{
		"request_id": requestID,
		"error":      body,
	})
}
