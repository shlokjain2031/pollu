package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockKashi imitates the kashi inference service. inferenceCalls counts
// /inference hits so dedup behaviour can be asserted.
type mockKashi struct {
	server         *httptest.Server
	inferenceCalls atomic.Int64
	failWith       atomic.Int32 // HTTP status to fail /inference with; 0 = succeed
	delay          time.Duration
}

func newMockKashi(t *testing.T) *mockKashi {
	mk := &mockKashi{}
	mux := http.NewServeMux()
	mux.HandleFunc("/meta", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"point_count": 65736, "model_version": "delhi_test", "feature_version": "v_test",
		})
	})
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	})
	mux.HandleFunc("/inference", func(w http.ResponseWriter, r *http.Request) {
		mk.inferenceCalls.Add(1)
		if mk.delay > 0 {
			time.Sleep(mk.delay)
		}
		if code := mk.failWith.Load(); code != 0 {
			w.WriteHeader(int(code))
			json.NewEncoder(w).Encode(map[string]string{"detail": "induced failure"})
			return
		}
		var req struct {
			Coords [][2]float64 `json:"coords"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		// Reject coordinates far outside Delhi, like the real service.
		for _, c := range req.Coords {
			if c[0] < 28 || c[0] > 29 || c[1] < 76 || c[1] > 78 {
				w.WriteHeader(404)
				json.NewEncoder(w).Encode(map[string]string{"detail": "outside the Delhi coverage area"})
				return
			}
		}
		preds := make([]float64, len(req.Coords))
		for i := range preds {
			preds[i] = 123.45
		}
		json.NewEncoder(w).Encode(map[string]any{"predictions": preds})
	})
	mk.server = httptest.NewServer(mux)
	t.Cleanup(mk.server.Close)
	return mk
}

func newTestCoordinator(t *testing.T, mk *mockKashi) *Coordinator {
	cfg := DefaultConfig()
	cfg.InferenceURL = mk.server.URL
	cfg.InferenceTimeout = 2 * time.Second
	cfg.RequestTimeout = 3 * time.Second
	// redis nil: tier skipped in tests
	return NewCoordinator(cfg, nil, NewInferenceClient(cfg))
}

func doPredict(t *testing.T, c *Coordinator, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewReader([]byte(body)))
	rec := httptest.NewRecorder()
	c.handlePredict(rec, req)
	var parsed map[string]any
	json.Unmarshal(rec.Body.Bytes(), &parsed)
	return rec, parsed
}

func TestPredictSuccess(t *testing.T) {
	c := newTestCoordinator(t, newMockKashi(t))
	rec, resp := doPredict(t, c, `{"lat":28.6315,"lon":77.2167,"timestamp_hour":"2026-06-09T14:00:00Z"}`)

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if resp["pm25"] != 123.45 {
		t.Errorf("want pm25=123.45, got %v", resp["pm25"])
	}
	if resp["source"] != "inference" {
		t.Errorf("want source=inference, got %v", resp["source"])
	}
	if resp["lat"] != 28.6315 || resp["lon"] != 77.2167 {
		t.Errorf("want echoed lat/lon, got %v/%v", resp["lat"], resp["lon"])
	}
}

func TestPredictCacheHitSecondCall(t *testing.T) {
	mk := newMockKashi(t)
	c := newTestCoordinator(t, mk)

	doPredict(t, c, `{"lat":28.60,"lon":77.10,"timestamp_hour":"2026-06-09T14:00:00Z"}`)
	rec, resp := doPredict(t, c, `{"lat":28.60,"lon":77.10,"timestamp_hour":"2026-06-09T14:00:00Z"}`)

	if rec.Code != 200 || resp["source"] != "cache" {
		t.Fatalf("want cached 200, got %d source=%v", rec.Code, resp["source"])
	}
	if mk.inferenceCalls.Load() != 1 {
		t.Errorf("want 1 inference call, got %d", mk.inferenceCalls.Load())
	}
}

func TestPredictCoordinateRounding(t *testing.T) {
	mk := newMockKashi(t)
	c := newTestCoordinator(t, mk)

	// Coordinates within ~1m round to the same canonical location → one call.
	for _, body := range []string{
		`{"lat":28.612345,"lon":77.201111,"timestamp_hour":"2026-06-09T14:00:00Z"}`,
		`{"lat":28.612346,"lon":77.201119,"timestamp_hour":"2026-06-09T14:00:00Z"}`,
		`{"lat":28.6123,"lon":77.2011,"timestamp_hour":"2026-06-09T14:00:00Z"}`,
	} {
		if rec, _ := doPredict(t, c, body); rec.Code != 200 {
			t.Fatalf("body %s: want 200, got %d", body, rec.Code)
		}
	}
	if mk.inferenceCalls.Load() != 1 {
		t.Errorf("want 1 inference call for coords sharing a 4-decimal key, got %d", mk.inferenceCalls.Load())
	}
}

func TestPredictTimestampNormalization(t *testing.T) {
	mk := newMockKashi(t)
	c := newTestCoordinator(t, mk)

	// Same hour expressed three ways must produce one inference call.
	for _, ts := range []string{
		"2026-06-09T14:00:00Z",
		"2026-06-09T14:37:22Z",
		"2026-06-09T19:30:00+05:30",
	} {
		rec, _ := doPredict(t, c, `{"lat":28.55,"lon":77.15,"timestamp_hour":"`+ts+`"}`)
		if rec.Code != 200 {
			t.Fatalf("ts %q: want 200, got %d: %s", ts, rec.Code, rec.Body.String())
		}
	}
	if mk.inferenceCalls.Load() != 1 {
		t.Errorf("want 1 inference call across equivalent timestamps, got %d", mk.inferenceCalls.Load())
	}
}

func TestPredictDedupConcurrent(t *testing.T) {
	mk := newMockKashi(t)
	mk.delay = 100 * time.Millisecond
	c := newTestCoordinator(t, mk)

	const n = 50
	var wg sync.WaitGroup
	codes := make([]int, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			rec, _ := doPredict(t, c, `{"lat":28.61,"lon":77.21,"timestamp_hour":"2026-06-09T14:00:00Z"}`)
			codes[i] = rec.Code
		}(i)
	}
	wg.Wait()

	for i, code := range codes {
		if code != 200 {
			t.Errorf("request %d: want 200, got %d", i, code)
		}
	}
	if calls := mk.inferenceCalls.Load(); calls != 1 {
		t.Errorf("want exactly 1 inference call for 50 concurrent identical requests, got %d", calls)
	}
}

func TestPredictValidation(t *testing.T) {
	c := newTestCoordinator(t, newMockKashi(t))

	cases := []struct {
		name string
		body string
		want int
	}{
		{"empty body", ``, 400},
		{"invalid json", `{not json`, 400},
		{"missing everything", `{}`, 400},
		{"missing timestamp", `{"lat":28.6,"lon":77.2}`, 400},
		{"bad timestamp", `{"lat":28.6,"lon":77.2,"timestamp_hour":"banana"}`, 400},
		{"year too old", `{"lat":28.6,"lon":77.2,"timestamp_hour":"1999-01-01T00:00:00Z"}`, 400},
		{"year too far", `{"lat":28.6,"lon":77.2,"timestamp_hour":"2099-01-01T00:00:00Z"}`, 400},
		{"lat without lon", `{"lat":28.6,"timestamp_hour":"2026-06-09T14:00:00Z"}`, 400},
		{"lon without lat", `{"lon":77.2,"timestamp_hour":"2026-06-09T14:00:00Z"}`, 400},
		{"lat out of range", `{"lat":91,"lon":77.2,"timestamp_hour":"2026-06-09T14:00:00Z"}`, 400},
		{"lon out of range", `{"lat":28.6,"lon":181,"timestamp_hour":"2026-06-09T14:00:00Z"}`, 400},
		{"outside coverage", `{"lat":19.0,"lon":72.8,"timestamp_hour":"2026-06-09T14:00:00Z"}`, 404},
	}
	for _, tc := range cases {
		rec, _ := doPredict(t, c, tc.body)
		if rec.Code != tc.want {
			t.Errorf("%s: want %d, got %d (%s)", tc.name, tc.want, rec.Code, rec.Body.String())
		}
	}
}

func TestPredictMethodNotAllowed(t *testing.T) {
	c := newTestCoordinator(t, newMockKashi(t))
	req := httptest.NewRequest(http.MethodGet, "/predict", nil)
	rec := httptest.NewRecorder()
	c.handlePredict(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("want 405, got %d", rec.Code)
	}
}

func TestPredictBodyTooLarge(t *testing.T) {
	c := newTestCoordinator(t, newMockKashi(t))
	huge := `{"lat":28.6,"lon":77.2,"pad":"` + strings.Repeat("x", maxRequestBodyBytes+1) + `"}`
	rec, _ := doPredict(t, c, huge)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("want 413, got %d", rec.Code)
	}
}

func TestPredictInferenceDown(t *testing.T) {
	mk := newMockKashi(t)
	c := newTestCoordinator(t, mk)
	mk.server.Close() // kill kashi

	rec, _ := doPredict(t, c, `{"lat":28.62,"lon":77.22,"timestamp_hour":"2026-06-09T14:00:00Z"}`)
	if rec.Code != 503 {
		t.Fatalf("want 503 when inference down, got %d: %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("want Retry-After header on 503")
	}
}

func TestPredictInferenceErrorCached(t *testing.T) {
	mk := newMockKashi(t)
	c := newTestCoordinator(t, mk)
	mk.failWith.Store(500)

	doPredict(t, c, `{"lat":28.63,"lon":77.23,"timestamp_hour":"2026-06-09T14:00:00Z"}`)
	callsAfterFirst := mk.inferenceCalls.Load()
	rec, _ := doPredict(t, c, `{"lat":28.63,"lon":77.23,"timestamp_hour":"2026-06-09T14:00:00Z"}`)

	if rec.Code != 503 {
		t.Fatalf("want 503, got %d", rec.Code)
	}
	if mk.inferenceCalls.Load() != callsAfterFirst {
		t.Errorf("second request should be served from the error cache, not hit kashi")
	}
}

func TestPredictRecoversAfterFailure(t *testing.T) {
	mk := newMockKashi(t)
	c := newTestCoordinator(t, mk)

	mk.failWith.Store(500)
	rec, _ := doPredict(t, c, `{"lat":28.64,"lon":77.24,"timestamp_hour":"2026-06-09T14:00:00Z"}`)
	if rec.Code != 503 {
		t.Fatalf("want 503 during outage, got %d", rec.Code)
	}

	mk.failWith.Store(0)
	// The ServiceUnavailable error is cached briefly; invalidate to simulate TTL expiry.
	model, feature, _ := c.versions()
	c.cache.Invalidate(CacheKey{Location: canonicalLocation(28.64, 77.24), TimestampHour: "2026-06-09T14:00:00Z", ModelVersion: model, FeatureVersion: feature})

	rec, resp := doPredict(t, c, `{"lat":28.64,"lon":77.24,"timestamp_hour":"2026-06-09T14:00:00Z"}`)
	if rec.Code != 200 {
		t.Fatalf("want 200 after recovery, got %d: %s", rec.Code, rec.Body.String())
	}
	if resp["pm25"] != 123.45 {
		t.Errorf("want pm25 after recovery, got %v", resp["pm25"])
	}
}

func TestCanonicalLocation(t *testing.T) {
	cases := map[[2]float64]string{
		{28.6315, 77.2167}:     "28.6315,77.2167",
		{28.612345, 77.201119}: "28.6123,77.2011",
		{28.0, 77.0}:           "28.0000,77.0000",
	}
	for in, want := range cases {
		if got := canonicalLocation(in[0], in[1]); got != want {
			t.Errorf("canonicalLocation(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestCanonicalHour(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"2026-06-09T14:37:22Z", "2026-06-09T14:00:00Z"},
		{"2026-06-09T19:30:00+05:30", "2026-06-09T14:00:00Z"},
		{"2026-06-09T14:00:00", "2026-06-09T14:00:00Z"},
	} {
		got, err := canonicalHour(tc.in)
		if err != nil {
			t.Errorf("canonicalHour(%q) error: %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("canonicalHour(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, bad := range []string{"", "banana", "2026-13-45T99:00:00Z", "1999-01-01T00:00:00Z"} {
		if _, err := canonicalHour(bad); err == nil {
			t.Errorf("canonicalHour(%q) should fail", bad)
		}
	}
}
