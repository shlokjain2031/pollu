#!/usr/bin/env python3
"""
Stress test: Measure coordinator behavior under increasing load
"""
import urllib.request
import json
import time
import concurrent.futures
import threading
import statistics

BASE_URL = "http://localhost:8080/predict"
TIMESTAMP = "2026-01-19T17:00:00Z"
GRIDS = ["g00491855", "g00491621", "g00491663"]  # Cached grids

request_count = 0
request_lock = threading.Lock()

def make_request(grid_id, request_num):
    """Make a single prediction request"""
    global request_count
    
    payload = json.dumps({
        'grid_id': grid_id,
        'timestamp_hour': TIMESTAMP
    }).encode('utf-8')
    
    req = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            latency = (time.time() - start) * 1000
            data = json.loads(response.read().decode('utf-8'))
            
            with request_lock:
                request_count += 1
            
            return {
                'success': True,
                'latency_ms': latency,
                'source': data.get('source'),
                'pm25': data.get('pm25')
            }
    except Exception as e:
        with request_lock:
            request_count += 1
        
        return {
            'success': False,
            'error': str(e)[:50],
            'latency_ms': (time.time() - start) * 1000
        }

def stress_test(num_requests, max_workers):
    """Run stress test with given concurrency"""
    global request_count
    request_count = 0
    
    grids = [GRIDS[i % len(GRIDS)] for i in range(num_requests)]
    
    print(f"\n{'='*70}")
    print(f"STRESS TEST: {num_requests} requests with {max_workers} max workers")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(make_request, grid, i)
            for i, grid in enumerate(grids)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    latencies = [r['latency_ms'] for r in successful]
    
    success_rate = (len(successful) / len(results)) * 100
    throughput = num_requests / total_time
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.0f} req/sec")
    print(f"Success: {len(successful)}/{len(results)} ({success_rate:.1f}%)")
    print(f"Failed: {len(failed)}")
    
    if latencies:
        print(f"Latency (ms):")
        print(f"  Min: {min(latencies):.1f}")
        print(f"  Max: {max(latencies):.1f}")
        print(f"  Avg: {statistics.mean(latencies):.1f}")
        print(f"  P50: {statistics.median(latencies):.1f}")
        if len(latencies) > 10:
            print(f"  P99: {sorted(latencies)[int(len(latencies)*0.99)]:.1f}")
    
    if failed:
        print(f"\nSample failures:")
        for f in failed[:3]:
            print(f"  - {f['error']}")
    
    return {
        'num_requests': num_requests,
        'max_workers': max_workers,
        'total_time': total_time,
        'throughput': throughput,
        'success_rate': success_rate,
        'success_count': len(successful),
        'failed_count': len(failed),
        'avg_latency': statistics.mean(latencies) if latencies else 0,
        'max_latency': max(latencies) if latencies else 0
    }

def main():
    """Run increasing load tests"""
    print("COORDINATOR STRESS TEST - Increasing Load")
    print("(Using cached grids from Redis)")
    
    test_configs = [
        (10, 10),
        (100, 50),
        (500, 100),
        (1000, 200),
        (5000, 500),
        (10000, 1000),
    ]
    
    results_summary = []
    
    for num_requests, max_workers in test_configs:
        try:
            result = stress_test(num_requests, max_workers)
            results_summary.append(result)
        except Exception as e:
            print(f"\n❌ FAILED at {num_requests} requests: {e}")
            break
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Requests':<12} {'Workers':<10} {'Throughput':<15} {'Success%':<12} {'Avg Lat(ms)':<12}")
    print(f"{'-'*70}")
    
    for r in results_summary:
        print(f"{r['num_requests']:<12} {r['max_workers']:<10} {r['throughput']:>6.0f} req/s    {r['success_rate']:>7.1f}%     {r['avg_latency']:>7.1f}ms")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    if results_summary:
        first = results_summary[0]
        last = results_summary[-1]
        
        throughput_change = (last['throughput'] / first['throughput'] - 1) * 100 if first['throughput'] > 0 else 0
        latency_change = (last['avg_latency'] / first['avg_latency'] - 1) * 100 if first['avg_latency'] > 0 else 0
        
        print(f"Load increase: {first['num_requests']} → {last['num_requests']} ({(last['num_requests']/first['num_requests']):.0f}×)")
        print(f"Throughput change: {throughput_change:+.0f}%")
        print(f"Latency change: {latency_change:+.0f}%")
        print(f"Final success rate: {last['success_rate']:.1f}%")
        
        if last['success_rate'] < 99:
            print(f"\n⚠️  System degradation detected at {last['num_requests']} concurrent requests")
        else:
            print(f"\n✅ System handling {last['num_requests']} requests with >99% success")

if __name__ == '__main__':
    main()
