#!/usr/bin/env python3
"""
Master Pipeline Orchestrator

Runs patliputra and nalanda pipelines sequentially for a given date range.
Manages workflow, logging, error handling, and dependency tracking.

Usage:
    python run_pipeline.py \
        --start-date 2024-01-01 \
        --end-date 2024-01-31 \
        --city mumbai \
        --verbose

    python run_pipeline.py \
        --start-date 2024-01-01 \
        --end-date 2024-01-31 \
        --skip-patliputra  # Run only nalanda

    python run_pipeline.py \
        --start-date 2024-01-01 \
        --end-date 2024-01-31 \
        --skip-nalanda  # Run only patliputra
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PipelineResult:
    """Result from running a single pipeline."""
    pipeline: str  # 'patliputra' or 'nalanda'
    status: str  # 'success', 'failed', 'skipped'
    start_time: str
    end_time: str
    duration_seconds: float
    returncode: int = 0
    stdout_lines: int = 0
    stderr_lines: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestrationResult:
    """Final result from orchestration run."""
    start_date: str
    end_date: str
    city: str
    total_duration_seconds: float
    start_time: str
    end_time: str
    patliputra_result: Optional[PipelineResult]
    nalanda_result: Optional[PipelineResult]
    overall_status: str  # 'success', 'partial_failure', 'failed'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "city": self.city,
            "total_duration_seconds": self.total_duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "patliputra_result": self.patliputra_result.to_dict() if self.patliputra_result else None,
            "nalanda_result": self.nalanda_result.to_dict() if self.nalanda_result else None,
            "overall_status": self.overall_status,
        }


# ============================================================================
# Pipeline Runners
# ============================================================================

def run_patliputra(
    start_date: str,
    end_date: str,
    city: str,
    target_row_idx: Optional[Path] = None,
    verbose: bool = False,
) -> PipelineResult:
    """Run patliputra pipeline for date range."""
    
    start_time = datetime.now().isoformat()
    print(f"\n{'='*80}")
    print(f"PATLIPUTRA PIPELINE")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"City: {city}")
    print(f"Start Time: {start_time}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "-m",
        "patliputra.pipeline",
        "run",
        "--start-date", start_date,
        "--end-date", end_date,
        "--city", city,
    ]
    
    # Add target-row-idx if provided
    if target_row_idx:
        cmd.extend(["--target-row-idx", str(target_row_idx)])
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None,  # No timeout; pipelines can run long
        )
        
        end_time = datetime.now().isoformat()
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
        
        stdout_lines = len(result.stdout.splitlines()) if result.stdout else 0
        stderr_lines = len(result.stderr.splitlines()) if result.stderr else 0
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr and verbose:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            status = "success"
            print(f"\n✓ Patliputra pipeline completed successfully")
        else:
            status = "failed"
            print(f"\n✗ Patliputra pipeline failed with return code {result.returncode}")
        
        return PipelineResult(
            pipeline="patliputra",
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            returncode=result.returncode,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            error_message=result.stderr if result.returncode != 0 else None,
        )
    
    except subprocess.TimeoutExpired:
        end_time = datetime.now().isoformat()
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
        print(f"\n✗ Patliputra pipeline timed out")
        return PipelineResult(
            pipeline="patliputra",
            status="failed",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            returncode=-1,
            error_message="Pipeline timed out",
        )
    
    except Exception as e:
        end_time = datetime.now().isoformat()
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
        print(f"\n✗ Patliputra pipeline error: {e}")
        return PipelineResult(
            pipeline="patliputra",
            status="failed",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            returncode=-1,
            error_message=str(e),
        )


def run_nalanda(
    start_date: str,
    end_date: str,
    verbose: bool = False,
) -> PipelineResult:
    """Run nalanda pipeline for date range."""
    
    start_time = datetime.now().isoformat()
    print(f"\n{'='*80}")
    print(f"NALANDA PIPELINE")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Start Time: {start_time}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "-m",
        "nalanda.cli",
        "assemble",
        "--start-date", start_date,
        "--end-date", end_date,
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None,
        )
        
        end_time = datetime.now().isoformat()
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
        
        stdout_lines = len(result.stdout.splitlines()) if result.stdout else 0
        stderr_lines = len(result.stderr.splitlines()) if result.stderr else 0
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr and verbose:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            status = "success"
            print(f"\n✓ Nalanda pipeline completed successfully")
        else:
            status = "failed"
            print(f"\n✗ Nalanda pipeline failed with return code {result.returncode}")
        
        return PipelineResult(
            pipeline="nalanda",
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            returncode=result.returncode,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            error_message=result.stderr if result.returncode != 0 else None,
        )
    
    except subprocess.TimeoutExpired:
        end_time = datetime.now().isoformat()
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
        print(f"\n✗ Nalanda pipeline timed out")
        return PipelineResult(
            pipeline="nalanda",
            status="failed",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            returncode=-1,
            error_message="Pipeline timed out",
        )
    
    except Exception as e:
        end_time = datetime.now().isoformat()
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()
        print(f"\n✗ Nalanda pipeline error: {e}")
        return PipelineResult(
            pipeline="nalanda",
            status="failed",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            returncode=-1,
            error_message=str(e),
        )


# ============================================================================
# Main Orchestration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline orchestrator for patliputra and nalanda",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--city",
        type=str,
        default="mumbai",
        help="City name for patliputra (default: mumbai)",
    )
    parser.add_argument(
        "--target-row-idx",
        type=Path,
        default=None,
        help="Path to target row_idx parquet file (optional)",
    )
    parser.add_argument(
        "--skip-patliputra",
        action="store_true",
        help="Skip patliputra pipeline and run only nalanda",
    )
    parser.add_argument(
        "--skip-nalanda",
        action="store_true",
        help="Skip nalanda pipeline and run only patliputra",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Save JSON summary of pipeline results to file",
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
        if start > end:
            print("✗ Start date must be before end date")
            sys.exit(1)
    except ValueError as e:
        print(f"✗ Invalid date format: {e}")
        sys.exit(1)
    
    # Initialize results
    orchestration_start = datetime.now()
    patliputra_result = None
    nalanda_result = None
    
    # Run patliputra
    if not args.skip_patliputra:
        patliputra_result = run_patliputra(
            start_date=args.start_date,
            end_date=args.end_date,
            city=args.city,
            target_row_idx=args.target_row_idx,
            verbose=args.verbose,
        )
        
        # Stop if patliputra fails and we didn't skip nalanda
        if patliputra_result.status == "failed" and not args.skip_nalanda:
            print("\n" + "!"*80)
            print("PATLIPUTRA FAILED - Stopping orchestration")
            print("!"*80)
            overall_status = "failed"
            nalanda_result = None
    else:
        print("\n→ Skipping patliputra pipeline (--skip-patliputra)")
    
    # Run nalanda
    if not args.skip_nalanda and (args.skip_patliputra or (patliputra_result and patliputra_result.status == "success")):
        nalanda_result = run_nalanda(
            start_date=args.start_date,
            end_date=args.end_date,
            verbose=args.verbose,
        )
    elif not args.skip_nalanda:
        print("\n→ Skipping nalanda pipeline (patliputra failed)")
    else:
        print("\n→ Skipping nalanda pipeline (--skip-nalanda)")
    
    # Determine overall status
    if patliputra_result and nalanda_result:
        if patliputra_result.status == "success" and nalanda_result.status == "success":
            overall_status = "success"
        elif patliputra_result.status == "success" or nalanda_result.status == "success":
            overall_status = "partial_failure"
        else:
            overall_status = "failed"
    elif patliputra_result:
        overall_status = patliputra_result.status
    elif nalanda_result:
        overall_status = nalanda_result.status
    else:
        overall_status = "skipped"
    
    # Create summary
    orchestration_end = datetime.now()
    total_duration = (orchestration_end - orchestration_start).total_seconds()
    
    result = OrchestrationResult(
        start_date=args.start_date,
        end_date=args.end_date,
        city=args.city,
        total_duration_seconds=total_duration,
        start_time=orchestration_start.isoformat(),
        end_time=orchestration_end.isoformat(),
        patliputra_result=patliputra_result,
        nalanda_result=nalanda_result,
        overall_status=overall_status,
    )
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(f"ORCHESTRATION COMPLETE")
    print(f"{'='*80}")
    print(f"Overall Status: {overall_status.upper()}")
    print(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    if patliputra_result:
        print(f"\nPatliputra: {patliputra_result.status.upper()} ({patliputra_result.duration_seconds:.1f}s)")
    else:
        print(f"\nPatliputra: SKIPPED")
    
    if nalanda_result:
        print(f"Nalanda: {nalanda_result.status.upper()} ({nalanda_result.duration_seconds:.1f}s)")
    else:
        print(f"Nalanda: SKIPPED")
    
    # Save summary if requested
    if args.output_summary:
        with open(args.output_summary, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSummary saved to: {args.output_summary}")
    
    print(f"{'='*80}\n")
    
    # Exit with appropriate code
    if overall_status == "success":
        sys.exit(0)
    elif overall_status == "partial_failure":
        sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
