#!/usr/bin/env python3

import urllib.request
import json
import concurrent.futures
import time

grids = ['g00491855', 'g00491621', 'g00491663', 'g00491876', 'g00491647'] * 2
timestamp = '2026-01-19T17:00:00Z'
base_url = 'http://localhost:8080/predict'

def make_request(grid, idx):
    start = time.time()
    try:
        payload = json.dumps({'grid_id': grid, 'timestamp_hour': timestamp}).encode('utf-8')
        req = urllib.request.Request(base_url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=10) as response:
            latency = (time.time() - start) * 1000
            data = json.loads(response.read().decode('utf-8'))
            pm25 = data.get('pm25', 'N/A')
            source = data.get('source', 'N/A')
            req_id = str(data.get('request_id', 'N/A'))[:8]
            return f'[{idx}] {grid} | PM25: {str(pm25):>6} | Source: {source:>8} | Latency: {latency:>6.1f}ms | Req: {req_id}'
    except Exception as e:
        return f'[{idx}] {grid} | FAILED: {str(e)}'

print('Testing 10 concurrent requests to coordinator...')
print()

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request, grid, idx) for idx, grid in enumerate(grids, 1)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

results.sort(key=lambda x: int(x.split(']')[0].split('[')[1]))
for result in results:
    print(result)

print()
print('All concurrent requests completed!')
