"""
Results management and storage for PEFT experiments.
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import sqlite3

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

# Handle imports for standalone usage
CONFIG_AVAILABLE = False
ExperimentConfig = None

try:
    from .standalone_config import ExperimentConfig
    CONFIG_AVAILABLE = True
except (ImportError, ValueError):
    # Create a dummy ExperimentConfig for type hints
    class ExperimentConfig:
        def to_dict(self):
            return {}
        
        @classmethod
        def from_dict(cls, data):
            return cls()
    
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    
    experiment_id: str
    config: "ExperimentConfig"
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Status and metadata
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    # Resource usage
    peak_memory_gb: float = 0.0
    average_memory_gb: float = 0.0
    
    # Training details
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get experiment duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def is_successful(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == "completed" and self.error_message is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result_dict = asdict(self)
        
        # Convert datetime objects to ISO strings
        if self.start_time:
            result_dict["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result_dict["end_time"] = self.end_time.isoformat()
        
        return result_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create result from dictionary."""
        # Convert ISO strings back to datetime
        if "start_time" in data and data["start_time"]:
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "end_time" in data and data["end_time"]:
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        
        # Reconstruct config if available
        if CONFIG_AVAILABLE and "config" in data:
            data["config"] = ExperimentConfig.from_dict(data["config"])
        
        return cls(**data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment result."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "peak_memory_gb": self.peak_memory_gb,
            "key_metrics": {
                k: v for k, v in self.metrics.items() 
                if k in ["final_accuracy", "final_loss", "best_accuracy"]
            },
            "model": self.config.model.name if self.config else "unknown",
            "dataset": self.config.dataset.name if self.config else "unknown"
        }


class ResultsManager:
    """
    Manager for storing and retrieving experiment results.
    
    Supports both file-based storage (JSON) and database storage (SQLite).
    """
    
    def __init__(
        self,
        storage_dir: Union[str, Path],
        use_database: bool = True,
        database_name: str = "experiments.db"
    ):
        """
        Initialize results manager.
        
        Args:
            storage_dir: Directory for storing results
            use_database: Whether to use SQLite database
            database_name: Name of SQLite database file
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_database = use_database
        self.database_path = self.storage_dir / database_name
        
        # Initialize database if enabled
        if self.use_database:
            self._init_database()
        
        logger.info(f"ResultsManager initialized with storage_dir: {self.storage_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for results storage."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Create experiments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id TEXT PRIMARY KEY,
                        config_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        start_time TEXT,
                        end_time TEXT,
                        duration_seconds REAL,
                        peak_memory_gb REAL,
                        error_message TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                        UNIQUE(experiment_id, metric_name)
                    )
                """)
                
                # Create indices for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_experiments_status 
                    ON experiments(status)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id 
                    ON metrics(experiment_id)
                """)
                
                conn.commit()
                
            logger.info(f"Database initialized at {self.database_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.use_database = False
    
    def add_result(self, result: ExperimentResult):
        """
        Add experiment result to storage.
        
        Args:
            result: ExperimentResult to store
        """
        # Save to JSON file
        self._save_json_result(result)
        
        # Save to database if enabled
        if self.use_database:
            self._save_database_result(result)
        
        logger.info(f"Result saved: {result.experiment_id}")
    
    def _save_json_result(self, result: ExperimentResult):
        """Save result to JSON file."""
        result_file = self.storage_dir / f"{result.experiment_id}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    def _save_database_result(self, result: ExperimentResult):
        """Save result to SQLite database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Insert or replace experiment record
                cursor.execute("""
                    INSERT OR REPLACE INTO experiments (
                        experiment_id, config_json, status, start_time, end_time,
                        duration_seconds, peak_memory_gb, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.experiment_id,
                    json.dumps(result.config.to_dict() if result.config else {}),
                    result.status,
                    result.start_time.isoformat() if result.start_time else None,
                    result.end_time.isoformat() if result.end_time else None,
                    result.duration_seconds,
                    result.peak_memory_gb,
                    result.error_message
                ))
                
                # Insert metrics
                for metric_name, metric_value in result.metrics.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO metrics (experiment_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """, (result.experiment_id, metric_name, float(metric_value)))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save result to database: {e}")
    
    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        Get experiment result by ID.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            ExperimentResult if found, None otherwise
        """
        # Try database first if available
        if self.use_database:
            result = self._get_database_result(experiment_id)
            if result:
                return result
        
        # Fallback to JSON file
        return self._get_json_result(experiment_id)
    
    def _get_json_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get result from JSON file."""
        result_file = self.storage_dir / f"{experiment_id}.json"
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            return ExperimentResult.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load result from JSON: {e}")
            return None
    
    def _get_database_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get result from database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get experiment record
                cursor.execute("""
                    SELECT config_json, status, start_time, end_time, 
                           duration_seconds, peak_memory_gb, error_message
                    FROM experiments WHERE experiment_id = ?
                """, (experiment_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                config_json, status, start_time, end_time, duration, peak_memory, error = row
                
                # Get metrics
                cursor.execute("""
                    SELECT metric_name, metric_value 
                    FROM metrics WHERE experiment_id = ?
                """, (experiment_id,))
                
                metrics = {name: value for name, value in cursor.fetchall()}
                
                # Reconstruct result
                config = None
                if CONFIG_AVAILABLE and config_json:
                    config = ExperimentConfig.from_dict(json.loads(config_json))
                
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    config=config,
                    metrics=metrics,
                    status=status,
                    start_time=datetime.fromisoformat(start_time) if start_time else None,
                    end_time=datetime.fromisoformat(end_time) if end_time else None,
                    peak_memory_gb=peak_memory or 0.0,
                    error_message=error
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get result from database: {e}")
            return None
    
    def list_results(
        self,
        status_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentResult]:
        """
        List experiment results with optional filtering.
        
        Args:
            status_filter: Filter by experiment status
            model_filter: Filter by model name
            dataset_filter: Filter by dataset name
            limit: Maximum number of results to return
            
        Returns:
            List of ExperimentResult objects
        """
        if self.use_database:
            return self._list_database_results(status_filter, model_filter, dataset_filter, limit)
        else:
            return self._list_json_results(status_filter, model_filter, dataset_filter, limit)
    
    def _list_json_results(
        self,
        status_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentResult]:
        """List results from JSON files."""
        results = []
        
        for result_file in self.storage_dir.glob("*.json"):
            if result_file.name == "summary.json":  # Skip summary files
                continue
            
            try:
                result = self._get_json_result(result_file.stem)
                if result is None:
                    continue
                
                # Apply filters
                if status_filter and result.status != status_filter:
                    continue
                
                if model_filter and result.config and result.config.model.name != model_filter:
                    continue
                
                if dataset_filter and result.config and result.config.dataset.name != dataset_filter:
                    continue
                
                results.append(result)
                
                # Apply limit
                if limit and len(results) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load result from {result_file}: {e}")
                continue
        
        return results
    
    def _list_database_results(
        self,
        status_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
        dataset_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExperimentResult]:
        """List results from database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Build query with filters
                query = "SELECT experiment_id FROM experiments WHERE 1=1"
                params = []
                
                if status_filter:
                    query += " AND status = ?"
                    params.append(status_filter)
                
                # For model/dataset filters, we'd need to parse JSON or add columns
                # For now, we'll filter after loading
                
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                experiment_ids = [row[0] for row in cursor.fetchall()]
                
                # Load full results
                results = []
                for exp_id in experiment_ids:
                    result = self._get_database_result(exp_id)
                    if result:
                        # Apply remaining filters
                        if model_filter and result.config and result.config.model.name != model_filter:
                            continue
                        if dataset_filter and result.config and result.config.dataset.name != dataset_filter:
                            continue
                        
                        results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to list results from database: {e}")
            return []
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all experiments."""
        results = self.list_results()
        
        if not results:
            return {"total_experiments": 0}
        
        # Basic counts
        total = len(results)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status == "failed")
        
        # Performance statistics for completed experiments
        completed_results = [r for r in results if r.status == "completed"]
        
        if completed_results:
            accuracies = [r.metrics.get("final_accuracy", 0) for r in completed_results]
            durations = [r.duration_seconds for r in completed_results]
            memory_usage = [r.peak_memory_gb for r in completed_results]
            
            stats = {
                "total_experiments": total,
                "completed": completed,
                "failed": failed,
                "success_rate": completed / total if total > 0 else 0,
                "performance": {
                    "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                    "max_accuracy": max(accuracies) if accuracies else 0,
                    "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
                    "avg_memory_gb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    "max_memory_gb": max(memory_usage) if memory_usage else 0
                }
            }
        else:
            stats = {
                "total_experiments": total,
                "completed": completed,
                "failed": failed,
                "success_rate": 0,
                "performance": {}
            }
        
        return stats
    
    def export_to_csv(self, output_path: Union[str, Path]):
        """Export results to CSV file."""
        results = self.list_results()
        
        if not results:
            logger.warning("No results to export")
            return
        
        # Prepare data for CSV
        rows = []
        for result in results:
            row = {
                "experiment_id": result.experiment_id,
                "status": result.status,
                "model": result.config.model.name if result.config else "unknown",
                "dataset": result.config.dataset.name if result.config else "unknown",
                "duration_seconds": result.duration_seconds,
                "peak_memory_gb": result.peak_memory_gb,
                "start_time": result.start_time.isoformat() if result.start_time else "",
                "end_time": result.end_time.isoformat() if result.end_time else "",
                "error_message": result.error_message or ""
            }
            
            # Add metrics as columns
            for metric_name, metric_value in result.metrics.items():
                row[f"metric_{metric_name}"] = metric_value
            
            rows.append(row)
        
        # Create DataFrame and save
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available - cannot export to CSV")
            return
        
        try:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Results exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report of all experiments."""
        stats = self.get_summary_statistics()
        results = self.list_results()
        
        report = []
        report.append("=" * 60)
        report.append("EXPERIMENT RESULTS SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        report.append(f"Total Experiments: {stats['total_experiments']}")
        report.append(f"Completed: {stats['completed']}")
        report.append(f"Failed: {stats['failed']}")
        report.append(f"Success Rate: {stats['success_rate']:.1%}")
        report.append("")
        
        # Performance statistics
        if "performance" in stats and stats["performance"]:
            perf = stats["performance"]
            report.append("Performance Statistics:")
            report.append(f"  Average Accuracy: {perf.get('avg_accuracy', 0):.3f}")
            report.append(f"  Best Accuracy: {perf.get('max_accuracy', 0):.3f}")
            report.append(f"  Average Duration: {perf.get('avg_duration_seconds', 0):.1f}s")
            report.append(f"  Average Memory: {perf.get('avg_memory_gb', 0):.1f}GB")
            report.append(f"  Peak Memory: {perf.get('max_memory_gb', 0):.1f}GB")
            report.append("")
        
        # Top performing experiments
        completed_results = [r for r in results if r.status == "completed"]
        if completed_results:
            # Sort by accuracy
            top_results = sorted(
                completed_results,
                key=lambda r: r.metrics.get("final_accuracy", 0),
                reverse=True
            )[:5]
            
            report.append("Top 5 Experiments by Accuracy:")
            for i, result in enumerate(top_results, 1):
                accuracy = result.metrics.get("final_accuracy", 0)
                model = result.config.model.name if result.config else "unknown"
                dataset = result.config.dataset.name if result.config else "unknown"
                report.append(f"  {i}. {result.experiment_id}")
                report.append(f"     Model: {model}, Dataset: {dataset}")
                report.append(f"     Accuracy: {accuracy:.3f}, Duration: {result.duration_seconds:.1f}s")
                report.append("")
        
        # Failed experiments
        failed_results = [r for r in results if r.status == "failed"]
        if failed_results:
            report.append(f"Failed Experiments ({len(failed_results)}):")
            for result in failed_results[:10]:  # Show first 10
                report.append(f"  - {result.experiment_id}: {result.error_message}")
            if len(failed_results) > 10:
                report.append(f"  ... and {len(failed_results) - 10} more")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def cleanup_old_results(self, days_old: int = 30):
        """Remove results older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0
        
        results = self.list_results()
        for result in results:
            if result.start_time and result.start_time < cutoff_date:
                # Remove JSON file
                json_file = self.storage_dir / f"{result.experiment_id}.json"
                if json_file.exists():
                    json_file.unlink()
                    removed_count += 1
                
                # Remove from database
                if self.use_database:
                    try:
                        with sqlite3.connect(self.database_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM metrics WHERE experiment_id = ?", 
                                         (result.experiment_id,))
                            cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", 
                                         (result.experiment_id,))
                            conn.commit()
                    except Exception as e:
                        logger.error(f"Failed to remove {result.experiment_id} from database: {e}")
        
        logger.info(f"Cleaned up {removed_count} old results")
        return removed_count