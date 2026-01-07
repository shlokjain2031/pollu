"""Report generation for model comparison.

This module generates comprehensive comparison reports in multiple formats:
- JSON: Machine-readable results for downstream tools
- Text: Human-readable summary for stakeholders
- Side-by-side comparison across persistence, linear, and LightGBM models

Usage:
    from dwarka.report import ReportGenerator, generate_full_report
    
    # Manual report building
    gen = ReportGenerator(output_dir=Path("dwarka/reports"))
    gen.add_model("persistence", "persistence", {}, results_persistence)
    gen.add_model("linear", "linear", {"alpha": 1.0}, results_linear)
    gen.add_model("lightgbm_v0", "lightgbm", lgb_params, results_lgbm)
    
    json_path = gen.generate_json_report()
    text_path = gen.generate_text_report()
    
    # Or use end-to-end helper
    json_path, text_path = generate_full_report(
        val_df=val_df,
        baselines=baselines,
        lgbm_model=model,
        lgbm_metadata=metadata,
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dwarka.evaluate import EvaluationResults


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelReport:
    """Complete report for a single model."""
    
    model_name: str
    model_type: str  # "persistence", "linear", "lightgbm"
    params: Dict[str, Any]
    evaluation: EvaluationResults
    passed_success_gate: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "params": self.params,
            "evaluation": self.evaluation.to_dict(),
            "passed_success_gate": self.passed_success_gate,
        }


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generate side-by-side comparison reports."""
    
    def __init__(
        self,
        output_dir: Path = Path("dwarka/reports"),
        r2_threshold: float = 0.65,
        mae_threshold: float = 18.0,
    ):
        """Initialize report generator.
        
        Args:
            output_dir: Where to save reports
            r2_threshold: Minimum R² for success gate
            mae_threshold: Maximum MAE for success gate
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.r2_threshold = r2_threshold
        self.mae_threshold = mae_threshold
        
        self.reports: List[ModelReport] = []
    
    def add_model(
        self,
        model_name: str,
        model_type: str,
        params: Dict[str, Any],
        evaluation: EvaluationResults,
    ) -> None:
        """Add a model's results to the report.
        
        Args:
            model_name: Model identifier
            model_type: Type of model ("persistence", "linear", "lightgbm")
            params: Model hyperparameters/config
            evaluation: EvaluationResults from evaluate_model()
        """
        passed = self._check_success_gate(evaluation)
        
        report = ModelReport(
            model_name=model_name,
            model_type=model_type,
            params=params,
            evaluation=evaluation,
            passed_success_gate=passed,
        )
        
        self.reports.append(report)
    
    def _check_success_gate(self, evaluation: EvaluationResults) -> bool:
        """Check if model passes success criteria.
        
        Success gate:
        - R² > threshold (default 0.65)
        - MAE < threshold (default 18.0 µg/m³)
        
        Args:
            evaluation: EvaluationResults to check
            
        Returns:
            True if model passes all criteria
        """
        return (
            evaluation.overall.r2 > self.r2_threshold
            and evaluation.overall.mae < self.mae_threshold
        )
    
    def generate_json_report(
        self,
        filename: str = "model_comparison.json",
    ) -> Path:
        """Save all results to JSON.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved JSON file
        """
        path = self.output_dir / filename
        
        data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "success_criteria": {
                "r2_threshold": self.r2_threshold,
                "mae_threshold": self.mae_threshold,
            },
            "models": [r.to_dict() for r in self.reports],
            "comparison": self._generate_comparison_summary(),
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        return path
    
    def generate_text_report(
        self,
        filename: str = "model_comparison.txt",
    ) -> Path:
        """Generate human-readable text report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved text file
        """
        path = self.output_dir / filename
        
        with open(path, "w") as f:
            self._write_header(f)
            self._write_overall_table(f)
            self._write_detailed_results(f)
            self._write_success_gate_assessment(f)
        
        return path
    
    def _write_header(self, f) -> None:
        """Write report header."""
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Number of models: {len(self.reports)}\n\n")
    
    def _write_overall_table(self, f) -> None:
        """Write overall metrics comparison table."""
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Huber':>8} {'Pass?':>6}\n")
        f.write("-" * 70 + "\n")
        
        for report in self.reports:
            e = report.evaluation.overall
            pass_str = "✓" if report.passed_success_gate else "✗"
            f.write(
                f"{report.model_name:<20} {e.mae:>8.2f} {e.rmse:>8.2f} "
                f"{e.r2:>8.3f} {e.huber_loss:>8.2f} {pass_str:>6}\n"
            )
        
        f.write("-" * 70 + "\n\n")
    
    def _write_detailed_results(self, f) -> None:
        """Write detailed stratified results for each model."""
        f.write("DETAILED RESULTS BY STRATUM\n")
        f.write("=" * 70 + "\n\n")
        
        for report in self.reports:
            f.write(f"{report.model_name.upper()}\n")
            f.write(f"  Type: {report.model_type}\n")
            f.write(f"  Overall: MAE={report.evaluation.overall.mae:.2f} µg/m³, "
                   f"R²={report.evaluation.overall.r2:.3f}\n\n")
            
            # PM Regime stratification
            if report.evaluation.by_pm_regime:
                f.write("  PM Regime:\n")
                for regime, metrics in report.evaluation.by_pm_regime.items():
                    f.write(
                        f"    {regime.capitalize():12s}: "
                        f"MAE={metrics.mae:>6.2f}, "
                        f"R²={metrics.r2:>6.3f}, "
                        f"n={metrics.n_samples:>7,} ({metrics.sample_pct:>5.1f}%)\n"
                    )
                f.write("\n")
            
            # Trust Quartile stratification
            if report.evaluation.by_trust_quartile:
                f.write("  Trust Quartile:\n")
                for q, metrics in report.evaluation.by_trust_quartile.items():
                    f.write(
                        f"    {q}: "
                        f"MAE={metrics.mae:>6.2f}, "
                        f"n={metrics.n_samples:>7,}\n"
                    )
                f.write("\n")
            
            # Season stratification
            if report.evaluation.by_season:
                f.write("  Season:\n")
                for season, metrics in report.evaluation.by_season.items():
                    f.write(
                        f"    {season.replace('_', ' ').capitalize():12s}: "
                        f"MAE={metrics.mae:>6.2f}, "
                        f"R²={metrics.r2:>6.3f}, "
                        f"n={metrics.n_samples:>7,}\n"
                    )
                f.write("\n")
            
            f.write("\n")
    
    def _write_success_gate_assessment(self, f) -> None:
        """Write success gate summary."""
        f.write("SUCCESS GATE ASSESSMENT\n")
        f.write("-" * 70 + "\n")
        f.write("Criteria:\n")
        f.write(f"  • R² > {self.r2_threshold}\n")
        f.write(f"  • MAE < {self.mae_threshold} µg/m³\n")
        f.write("  • Beats persistence baseline\n\n")
        
        passed_models = [r for r in self.reports if r.passed_success_gate]
        
        if passed_models:
            f.write("Passed Models:\n")
            for report in passed_models:
                f.write(
                    f"  ✓ {report.model_name}: "
                    f"R²={report.evaluation.overall.r2:.3f}, "
                    f"MAE={report.evaluation.overall.mae:.2f}\n"
                )
            
            # Find best by R²
            best = max(passed_models, key=lambda r: r.evaluation.overall.r2)
            f.write(f"\nRECOMMENDED MODEL: {best.model_name}\n")
        else:
            f.write("⚠ No models passed success gate\n")
            f.write("\nAll models:\n")
            for report in self.reports:
                f.write(
                    f"  ✗ {report.model_name}: "
                    f"R²={report.evaluation.overall.r2:.3f}, "
                    f"MAE={report.evaluation.overall.mae:.2f}\n"
                )
    
    def _generate_comparison_summary(self) -> Dict[str, Any]:
        """Generate high-level comparison summary."""
        if not self.reports:
            return {}
        
        # Find best by R²
        best_r2 = max(self.reports, key=lambda r: r.evaluation.overall.r2)
        
        # Find best by MAE
        best_mae = min(self.reports, key=lambda r: r.evaluation.overall.mae)
        
        # Find which beat persistence
        persistence = next(
            (r for r in self.reports if r.model_type == "persistence"),
            None
        )
        
        beats_persistence = []
        if persistence:
            for r in self.reports:
                if r.evaluation.overall.r2 > persistence.evaluation.overall.r2:
                    beats_persistence.append(r.model_name)
        
        return {
            "best_by_r2": {
                "model": best_r2.model_name,
                "r2": best_r2.evaluation.overall.r2,
            },
            "best_by_mae": {
                "model": best_mae.model_name,
                "mae": best_mae.evaluation.overall.mae,
            },
            "beats_persistence": beats_persistence,
            "passed_success_gate": [
                r.model_name for r in self.reports
                if r.passed_success_gate
            ],
        }


# =============================================================================
# End-to-End Report Generation
# =============================================================================

def generate_full_report(
    val_df: pd.DataFrame,
    baselines: Dict[str, Any],
    lgbm_model: Any,
    lgbm_metadata: Dict[str, Any],
    schema_path: Path = Path("artifacts/v1/features/feature_schema.json"),
    output_dir: Path = Path("dwarka/reports"),
) -> Tuple[Path, Path]:
    """End-to-end report generation for all models.
    
    Args:
        val_df: Validation dataframe (with features, target, metadata)
        baselines: Dict returned from baselines.load_baselines()
        lgbm_model: Loaded LightGBM Booster
        lgbm_metadata: Metadata dict from training
        schema_path: Path to frozen schema
        output_dir: Output directory for reports
        
    Returns:
        Tuple of (json_path, text_path)
    """
    from nalanda.feature_freeze import FrozenFeatureSchema
    from dwarka.evaluate import evaluate_model
    
    # Load schema
    schema = FrozenFeatureSchema.load(schema_path)
    
    # Extract data
    X = schema.extract_X(val_df)
    y = val_df["pm25_t"].values
    timestamps = val_df["timestamp"] if "timestamp" in val_df.columns else None
    trust_weights = val_df["target_trust"].values if "target_trust" in val_df.columns else None
    
    # Get predictions
    persistence_pred = baselines["persistence"].predict(val_df)
    linear_pred = baselines["linear"].predict(X)
    lgbm_pred = lgbm_model.predict(X)
    
    # Evaluate all models
    results_persistence = evaluate_model(
        y, persistence_pred, "persistence_lag24h", timestamps, trust_weights
    )
    
    results_linear = evaluate_model(
        y, linear_pred, "ridge_regression", timestamps, trust_weights
    )
    
    results_lgbm = evaluate_model(
        y, lgbm_pred,
        lgbm_metadata.get("model_version", "lightgbm_v0"),
        timestamps, trust_weights
    )
    
    # Generate reports
    gen = ReportGenerator(output_dir)
    
    gen.add_model(
        "persistence_lag24h",
        "persistence",
        baselines["metadata"]["persistence"]["params"],
        results_persistence,
    )
    
    gen.add_model(
        "ridge_regression",
        "linear",
        baselines["metadata"]["linear"]["params"],
        results_linear,
    )
    
    gen.add_model(
        lgbm_metadata.get("model_version", "lightgbm_v0"),
        "lightgbm",
        lgbm_metadata.get("lgb_params", {}),
        results_lgbm,
    )
    
    # Generate both formats
    json_path = gen.generate_json_report()
    text_path = gen.generate_text_report()
    
    return json_path, text_path
