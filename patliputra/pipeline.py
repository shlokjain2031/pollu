#!/usr/bin/env python3
"""
Patliputra Data Pipeline Runner

Orchestrates all data fetching and processing scripts in the correct order
with dependency management and error handling.

Pipeline Stages:
    1. Grid Creation (prerequisite for all)
    2. Meteorological Data (OpenMeteo) - SEQUENTIAL (API rate limits)
    3. Satellite Data - SEQUENTIAL (Earth Engine quotas)
       - Landsat8 (Earth Engine)
       - Sentinel2 (Earth Engine)
    4. Air Quality Ground Truth (CPCB) - INDEPENDENT

Usage:
    # Run full pipeline for a date range
    python -m patliputra.pipeline run \
        --start-date 2018-01-01 \
        --end-date 2018-12-31 \
        --city mumbai
    
    # Run specific stages
    python -m patliputra.pipeline run \
        --stages meteo,satellite \
        --start-date 2021-06-01 \
        --end-date 2021-06-30
    
    # Check status of artifacts
    python -m patliputra.pipeline status --city mumbai
    
    # Validate artifacts
    python -m patliputra.pipeline validate --start-date 2018-01-01 --end-date 2018-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from patliputra.logging_utils import setup_logging


class Stage(str, Enum):
    """Pipeline stages in dependency order."""
    GRID = "grid"
    COMPUTE_TARGET_ROW_IDXS = "compute_target_row_idxs"
    METEO = "meteo"
    LANDSAT8 = "landsat8"
    SENTINEL2 = "sentinel2"
    AQI = "aqi"


# Stage dependencies (must run before)
STAGE_DEPENDENCIES = {
    Stage.GRID: [],
    Stage.COMPUTE_TARGET_ROW_IDXS: [Stage.GRID],
    Stage.METEO: [Stage.GRID, Stage.COMPUTE_TARGET_ROW_IDXS],
    Stage.LANDSAT8: [Stage.GRID, Stage.COMPUTE_TARGET_ROW_IDXS],
    Stage.SENTINEL2: [Stage.GRID, Stage.COMPUTE_TARGET_ROW_IDXS],
    Stage.AQI: [],  # Independent, uses raw CSV data
}


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: Stage
    script: str
    concurrent_safe: bool  # Can run concurrently with other stages?
    rate_limited: bool  # Has API/quota rate limits?
    description: str
    required_args: List[str] = field(default_factory=list)
    optional_args: List[str] = field(default_factory=list)


# Stage configurations
STAGE_CONFIGS = {
    Stage.GRID: StageConfig(
        name=Stage.GRID,
        script="patliputra/create_grid.py",
        concurrent_safe=True,
        rate_limited=False,
        description="Generate 30m grid for city",
        required_args=["--grid-id", "--grid-version"],
        optional_args=["--bounds"],
    ),
    Stage.COMPUTE_TARGET_ROW_IDXS: StageConfig(
        name=Stage.COMPUTE_TARGET_ROW_IDXS,
        script="patliputra/compute_target_row_idxs.py",
        concurrent_safe=True,
        rate_limited=False,
        description="Compute target row_idxs from sensor influence zones",
        required_args=["--city"],
        optional_args=["--bbox", "--grid-artifact", "--radius-m", "--output", "--openaq-token"],
    ),
    Stage.METEO: StageConfig(
        name=Stage.METEO,
        script="patliputra/openmeteo_signals.py",
        concurrent_safe=False,  # API rate limits
        rate_limited=True,
        description="Fetch meteorological data from Open-Meteo",
        required_args=["--grid-artifact", "--city", "--start-date", "--end-date"],
        optional_args=["--hourly", "--rate-limit"],
    ),
    Stage.LANDSAT8: StageConfig(
        name=Stage.LANDSAT8,
        script="patliputra/landsat8_signals.py",
        concurrent_safe=False,  # Earth Engine quotas
        rate_limited=True,
        description="Fetch Landsat-8 satellite data via Earth Engine",
        required_args=["--grid-artifact", "--city", "--start-date", "--end-date", "--project"],
        optional_args=["--batch-size", "--max-workers", "--target-row-idx"],
    ),
    Stage.SENTINEL2: StageConfig(
        name=Stage.SENTINEL2,
        script="patliputra/sentinel2_signals.py",
        concurrent_safe=False,  # Earth Engine quotas
        rate_limited=True,
        description="Fetch Sentinel-2 satellite data via Earth Engine",
        required_args=["--grid-artifact", "--city", "--start-date", "--end-date", "--project"],
        optional_args=["--workers"],
    ),
    Stage.AQI: StageConfig(
        name=Stage.AQI,
        script="patliputra/cpcb_aqi.py",
        concurrent_safe=True,  # Local CSV processing
        rate_limited=False,
        description="Process CPCB AQI ground truth data",
        required_args=[],
        optional_args=["--input-dir", "--force"],
    ),
}


@dataclass
class StageResult:
    """Result of running a pipeline stage."""
    stage: Stage
    success: bool
    duration_seconds: float
    message: str
    artifacts_created: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of running the full pipeline."""
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    stages_run: List[Stage]
    stages_succeeded: List[Stage]
    stages_failed: List[Stage]
    stage_results: Dict[Stage, StageResult]
    
    @property
    def success(self) -> bool:
        return len(self.stages_failed) == 0
    
    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "stages_run": [s.value for s in self.stages_run],
            "stages_succeeded": [s.value for s in self.stages_succeeded],
            "stages_failed": [s.value for s in self.stages_failed],
            "stage_results": {
                s.value: {
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "message": r.message,
                    "artifacts_created": r.artifacts_created,
                    "error": r.error,
                }
                for s, r in self.stage_results.items()
            },
        }


class PipelineRunner:
    """Orchestrates the data pipeline."""
    
    def __init__(
        self,
        city: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        grid_artifact: Optional[str] = None,
        ee_project: Optional[str] = None,
        openaq_token: Optional[str] = None,
        hourly: bool = False,
        verbose: bool = False,
    ):
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        self.grid_artifact = grid_artifact or f"artifacts/v1/grids/{city}/grid.parquet"
        self.ee_project = ee_project
        self.openaq_token = openaq_token
        self.hourly = hourly
        self.verbose = verbose
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.artifacts_root = self.project_root / "artifacts" / "v1"
    
    def run_stage(self, stage: Stage, **kwargs) -> StageResult:
        """Run a single pipeline stage."""
        config = STAGE_CONFIGS[stage]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Running stage: {stage.value.upper()}")
            print(f"Description: {config.description}")
            print(f"Rate limited: {config.rate_limited}")
            print(f"{'='*80}\n")
        
        # Build command. If the configured script is a path (contains '/'
        # or ends with '.py'), run it as a module using `-m` so package
        # imports resolve correctly when executed from the project root.
        script = config.script
        if script.endswith(".py") or "/" in script:
            module = script.replace("/", ".")
            if module.endswith(".py"):
                module = module[:-3]
            cmd = [sys.executable, "-m", module]
        else:
            # assume already a module path
            cmd = [sys.executable, "-m", script]
        
        # Add common arguments
        if stage == Stage.COMPUTE_TARGET_ROW_IDXS:
            cmd.extend(["--city", self.city])
            cmd.extend(["--grid-artifact", self.grid_artifact])
            if self.openaq_token:
                cmd.extend(["--openaq-token", self.openaq_token])
        
        if stage in [Stage.METEO, Stage.LANDSAT8, Stage.SENTINEL2]:
            cmd.extend(["--grid-artifact", self.grid_artifact])
            cmd.extend(["--city", self.city])
        
        if stage in [Stage.METEO, Stage.LANDSAT8, Stage.SENTINEL2] and self.start_date and self.end_date:
            cmd.extend(["--start-date", self.start_date])
            cmd.extend(["--end-date", self.end_date])
        
        if stage in [Stage.LANDSAT8, Stage.SENTINEL2] and self.ee_project:
            cmd.extend(["--project", self.ee_project])
        
        if stage == Stage.METEO and self.hourly:
            cmd.append("--hourly")
        
        # Add stage-specific kwargs
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
        
        # Run command
        start_time = datetime.now()
        try:
            if self.verbose:
                print(f"Command: {' '.join(cmd)}\n")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=not self.verbose,
                text=True,
                check=True,
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return StageResult(
                stage=stage,
                success=True,
                duration_seconds=duration,
                message=f"Stage {stage.value} completed successfully",
            )
        
        except subprocess.CalledProcessError as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = e.stderr if e.stderr else str(e)
            
            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=duration,
                message=f"Stage {stage.value} failed",
                error=error_msg,
            )
    
    def check_dependencies(self, stage: Stage) -> List[Stage]:
        """Check if stage dependencies are satisfied."""
        missing = []
        for dep in STAGE_DEPENDENCIES[stage]:
            # Check if dependency artifacts exist
            if dep == Stage.GRID:
                if not Path(self.grid_artifact).exists():
                    missing.append(dep)
        return missing
    
    def run_pipeline(
        self,
        stages: Optional[List[Stage]] = None,
        skip_dependencies: bool = False,
    ) -> PipelineResult:
        """
        Run the full pipeline or selected stages.
        
        Args:
            stages: List of stages to run. If None, runs all stages.
            skip_dependencies: Skip dependency checks (use with caution).
        
        Returns:
            PipelineResult with execution details.
        """
        start_time = datetime.now()
        
        # Determine stages to run
        if stages is None:
            stages = list(Stage)
        
        # Sort stages by dependency order
        stages_ordered = self._topological_sort(stages)
        
        # Results tracking
        stage_results: Dict[Stage, StageResult] = {}
        succeeded: List[Stage] = []
        failed: List[Stage] = []
        
        for stage in stages_ordered:
            # Check dependencies
            if not skip_dependencies:
                missing_deps = self.check_dependencies(stage)
                if missing_deps:
                    result = StageResult(
                        stage=stage,
                        success=False,
                        duration_seconds=0,
                        message=f"Missing dependencies: {[d.value for d in missing_deps]}",
                        error="Dependencies not satisfied",
                    )
                    stage_results[stage] = result
                    failed.append(stage)
                    continue
            
            # Run stage
            result = self.run_stage(stage)
            stage_results[stage] = result
            
            if result.success:
                succeeded.append(stage)
                if self.verbose:
                    print(f"✓ {stage.value} completed in {result.duration_seconds:.1f}s")
            else:
                failed.append(stage)
                if self.verbose:
                    print(f"✗ {stage.value} failed: {result.error}")
                
                # Stop on failure (fail-fast)
                break
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return PipelineResult(
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            stages_run=stages_ordered,
            stages_succeeded=succeeded,
            stages_failed=failed,
            stage_results=stage_results,
        )
    
    def _topological_sort(self, stages: List[Stage]) -> List[Stage]:
        """Sort stages in dependency order."""
        # Simple topological sort
        sorted_stages = []
        visited = set()
        
        def visit(stage: Stage):
            if stage in visited:
                return
            visited.add(stage)
            
            for dep in STAGE_DEPENDENCIES[stage]:
                if dep in stages:
                    visit(dep)
            
            if stage in stages:
                sorted_stages.append(stage)
        
        for stage in stages:
            visit(stage)
        
        return sorted_stages
    
    def status(self) -> Dict[str, any]:
        """Check status of pipeline artifacts."""
        status = {}
        
        # Check grid
        grid_path = Path(self.grid_artifact)
        status["grid"] = {
            "exists": grid_path.exists(),
            "path": str(grid_path),
        }
        
        # Check meteo
        meteo_dir = self.artifacts_root / "openmeteo" / self.city
        if meteo_dir.exists():
            meteo_files = list(meteo_dir.glob("*.parquet"))
            status["meteo"] = {
                "exists": True,
                "file_count": len(meteo_files),
                "path": str(meteo_dir),
            }
        else:
            status["meteo"] = {"exists": False}
        
        # Check landsat8
        landsat_dir = self.artifacts_root / "landsat8" / self.city
        if landsat_dir.exists():
            landsat_files = list(landsat_dir.glob("*.parquet"))
            status["landsat8"] = {
                "exists": True,
                "file_count": len(landsat_files),
                "path": str(landsat_dir),
            }
        else:
            status["landsat8"] = {"exists": False}
        
        # Check sentinel2
        sentinel_dir = self.artifacts_root / "sentinel2" / self.city
        if sentinel_dir.exists():
            sentinel_files = list(sentinel_dir.glob("*.parquet"))
            status["sentinel2"] = {
                "exists": True,
                "file_count": len(sentinel_files),
                "path": str(sentinel_dir),
            }
        else:
            status["sentinel2"] = {"exists": False}
        
        # Check AQI
        aqi_dir = self.artifacts_root / "aqi"
        if aqi_dir.exists():
            aqi_files = list(aqi_dir.glob("*.parquet"))
            status["aqi"] = {
                "exists": True,
                "file_count": len(aqi_files),
                "path": str(aqi_dir),
            }
        else:
            status["aqi"] = {"exists": False}
        
        return status


def main():
    # Initialize logging
    logger = setup_logging("pipeline")
    
    parser = argparse.ArgumentParser(
        description="Patliputra data pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline stages")
    run_parser.add_argument(
        "--city",
        required=True,
        help="City name (e.g., mumbai)",
    )
    run_parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    run_parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    run_parser.add_argument(
        "--stages",
        help="Comma-separated list of stages to run (default: all)",
    )
    run_parser.add_argument(
        "--grid-artifact",
        help="Path to grid artifact (default: artifacts/v1/grids/{city}/grid.parquet)",
    )
    run_parser.add_argument(
        "--ee-project",
        required=True,
        help="Earth Engine project ID (required for satellite stages)",
    )
    run_parser.add_argument(
        "--openaq-token",
        required=True,
        help="OpenAQ API token (for compute_target_row_idxs stage)",
    )
    run_parser.add_argument(
        "--hourly",
        action="store_true",
        help="Fetch hourly meteo data (default: daily snapshots)",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    run_parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="Skip dependency checks (use with caution)",
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check pipeline artifact status")
    status_parser.add_argument(
        "--city",
        required=True,
        help="City name (e.g., mumbai)",
    )
    status_parser.add_argument(
        "--grid-artifact",
        help="Path to grid artifact",
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate pipeline artifacts")
    validate_parser.add_argument(
        "--city",
        required=True,
        help="City name",
    )
    validate_parser.add_argument(
        "--start-date",
        help="Start date for validation",
    )
    validate_parser.add_argument(
        "--end-date",
        help="End date for validation",
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Parse stages
        stages = None
        if args.stages:
            stage_names = [s.strip() for s in args.stages.split(",")]
            stages = [Stage(name) for name in stage_names]
        
        # Create runner
        runner = PipelineRunner(
            city=args.city,
            start_date=args.start_date,
            end_date=args.end_date,
            grid_artifact=args.grid_artifact,
            ee_project=args.ee_project,
            openaq_token=args.openaq_token,
            hourly=args.hourly,
            verbose=args.verbose,
        )
        
        # Run pipeline
        result = runner.run_pipeline(stages, skip_dependencies=args.skip_dependencies)
        
        # Print results
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Stages run: {len(result.stages_run)}")
        print(f"Succeeded: {len(result.stages_succeeded)}")
        print(f"Failed: {len(result.stages_failed)}")
        
        if result.success:
            print("\n✓ Pipeline completed successfully")
        else:
            print(f"\n✗ Pipeline failed at stage: {result.stages_failed[0].value}")
            failed_result = result.stage_results[result.stages_failed[0]]
            print(f"Error: {failed_result.error}")
            sys.exit(1)
    
    elif args.command == "status":
        runner = PipelineRunner(
            city=args.city,
            grid_artifact=args.grid_artifact,
        )
        
        status = runner.status()
        print(json.dumps(status, indent=2))
    
    elif args.command == "validate":
        print("Validation not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
