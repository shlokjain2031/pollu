# Migrate all Landsat parquet files to new grid schema
# This script processes each file individually with error handling

$ErrorActionPreference = "Continue"

# Find all Landsat signal parquet files
$files = Get-ChildItem -Path "tmp\landsat8_*_signals.backup.parquet" | Select-Object -ExpandProperty FullName

Write-Host "Found $($files.Count) Landsat parquet files to migrate`n" -ForegroundColor Cyan

# Counters
$total = $files.Count
$current = 0
$success = 0
$skipped = 0
$failed = 0

# Process each file
foreach ($file in $files) {
    $current++
    $fileName = Split-Path $file -Leaf
    
    Write-Host "[$current/$total] Processing: $fileName" -ForegroundColor Yellow
    
    # Run migration script
    $output = & .\.venv\Scripts\python.exe scripts\migrate_old_parquets.py --input $file 2>&1
    
    # Check result
    if ($LASTEXITCODE -eq 0) {
        if ($output -match "Already uses new schema") {
            $skipped++
            Write-Host "  -> Skipped (already migrated)" -ForegroundColor Gray
        } else {
            $success++
            Write-Host "  -> Success" -ForegroundColor Green
        }
    } else {
        $failed++
        Write-Host "  -> Failed" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
    }
    
    Write-Host ""
}

# Summary
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "MIGRATION SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Total files:    $total" -ForegroundColor White
Write-Host "Success:        $success" -ForegroundColor Green
Write-Host "Skipped:        $skipped" -ForegroundColor Gray
Write-Host "Failed:         $failed" -ForegroundColor Red

if ($failed -eq 0) {
    Write-Host "`nAll files migrated successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nSome files failed to migrate. Check errors above." -ForegroundColor Red
    exit 1
}
