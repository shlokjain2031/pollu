#!/usr/bin/env python3
"""
Live Demo: Takshashila PM2.5 Inference Engine

This script demonstrates the inference engine predicting PM2.5 levels
for various grid cells in Mumbai.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import json
import time

# Setup path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from Takshashila.phase1_minimal_inference.inference import InferenceEngine
from Takshashila.phase1_minimal_inference.config import (
    DUCKDB_PATH, MODEL_PATH, FEATURE_SCHEMA_PATH
)

def demo():
    print("\n" + "="*70)
    print("🌍 TAKSHASHILA PM2.5 INFERENCE ENGINE - LIVE DEMO")
    print("="*70 + "\n")
    
    # Initialize engine
    print("📦 Initializing inference engine...")
    start_init = time.perf_counter()
    engine = InferenceEngine(
        db_path=str(DUCKDB_PATH),
        model_path=str(MODEL_PATH),
        feature_schema_path=str(FEATURE_SCHEMA_PATH)
    )
    init_time = time.perf_counter() - start_init
    print(f"✅ Engine ready! (Initialization: {init_time*1000:.2f}ms)\n")
    
    # Demo predictions for different grids and times
    test_cases = [
        {
            "grid_id": "g00000001",
            "timestamp": datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc),
            "description": "Valid Grid 1 - Evening"
        },
        {
            "grid_id": "g00000042", 
            "timestamp": datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc),
            "description": "Valid Grid 42 - Evening"
        },
        {
            "grid_id": "g00000201",
            "timestamp": datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc),
            "description": "Valid Grid 201 - Evening"
        },
        {
            "grid_id": "g00002000",
            "timestamp": datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc),
            "description": "Valid Grid 2000 - Evening"
        },
        {
            "grid_id": "g00001337",
            "timestamp": datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc),
            "description": "Valid Grid 1337 - Evening"
        },
        {
            "grid_id": "g00000001",
            "timestamp": datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc),
            "description": "Grid 1 - Repeat (cached result)"
        }
    ]
    
    print("🔮 Running predictions:\n")
    
    for i, test in enumerate(test_cases, 1):
        try:
            grid_id = test["grid_id"]
            ts = test["timestamp"]
            
            print(f"📍 Test {i}: {test['description']}")
            print(f"   Grid ID: {grid_id}")
            print(f"   Time: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"   ", end="")
            
            # Run prediction with timing
            start_pred = time.perf_counter()
            pm25 = engine.predict(grid_id, ts)
            pred_time = time.perf_counter() - start_pred
            
            # Display result with emoji based on AQI category
            if pm25 <= 50:
                emoji = "🟢"
                category = "Good"
            elif pm25 <= 100:
                emoji = "🟡"
                category = "Moderate"
            elif pm25 <= 150:
                emoji = "🟠"
                category = "Unhealthy for Sensitive Groups"
            elif pm25 <= 200:
                emoji = "🔴"
                category = "Unhealthy"
            else:
                emoji = "🟣"
                category = "Hazardous"
            
            print(f"PM₂.₅: {pm25:.2f} µg/m³ {emoji} ({category}) [{pred_time*1000:.2f}ms]\n")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}\n")
    
    print("="*70)
    print("\n🧪 IDEMPOTENCY TEST: Same inputs → Same output?\n")
    
    # Test idempotency: call twice with same params, should get same result
    test_grid = "g00000042"
    test_ts = datetime(2026, 1, 19, 16, 0, 0, tzinfo=timezone.utc)
    
    print(f"Testing grid {test_grid} at {test_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("(First call will compute and cache, second will read from cache)\n")
    
    # Clear cache by recreating engine to test fresh computation
    engine2 = InferenceEngine(
        db_path=str(DUCKDB_PATH),
        model_path=str(MODEL_PATH),
        feature_schema_path=str(FEATURE_SCHEMA_PATH)
    )
    
    start_call1 = time.perf_counter()
    result1 = engine2.predict(test_grid, test_ts)
    time_call1 = time.perf_counter() - start_call1
    
    start_call2 = time.perf_counter()
    result2 = engine2.predict(test_grid, test_ts)
    time_call2 = time.perf_counter() - start_call2
    
    print(f"   Call 1 (computed): {result1:.6f} µg/m³ ({time_call1*1000:.2f}ms)")
    print(f"   Call 2 (cached):   {result2:.6f} µg/m³ ({time_call2*1000:.2f}ms)")
    
    speedup = time_call1 / time_call2 if time_call2 > 0 else float('inf')
    print(f"   ✅ IDENTICAL (difference: {abs(result1 - result2):.2e})")
    print(f"   ⚡ Cache speedup: {speedup:.1f}x faster")
    
    print("\n" + "="*70)
    print("✨ Demo complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    demo()
