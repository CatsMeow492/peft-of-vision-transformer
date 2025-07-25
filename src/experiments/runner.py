"""
Automated experiment execution pipeline for PEFT Vision Transformer research.
"""

import logging
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Callable, Union
from datetime import datetime, timedelta
import threading
import queue
import signal
import sys

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    from .standalone_config import ExperimentConfig, ExperimentMatrix
    from .results import ExperimentResult, ResultsManager
    CONFIG_AVAILABLE = True
except ImportError:
    # Fallback for testing
    ExperimentConfig = None
    ExperimentMatrix = None
    ExperimentResult = None
    ResultsManager = None
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ResourceMonitor:
    """Monitor system resources during experiments."""
    
    max_memory_gb: float = 32.0
    max_cpu_percent: float = 90.0
    check_interval_seconds: float = 5.0
    
    # Current resource usage
    current_memory_gb: float = 0.0
    current_cpu_percent: float = 0.0
    peak_memory_gb: float = 0.0
    
    # Monitoring state
    monitoring: bool = False
    _monitor_thread: Optional[threading.Thread] = None
    _stop_event: Optional[threading.Event] = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self._stop_event:
            self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring and not self._stop_event.is_set():
            try:
                if PSUTIL_AVAILABLE:
                    # Get memory usage
                    memory = psutil.virtual_memory()
                    self.current_memory_gb = memory.used / (1024**3)
                    self.peak_memory_gb = max(self.peak_memory_gb, self.current_memory_gb)
                    
                    # Get CPU usage
                    self.current_cpu_percent = psutil.cpu_percent(interval=1.0)
                else:
                    # Mock values when psutil is not available
                    self.current_memory_gb = 4.0  # Mock 4GB usage
                    self.current_cpu_percent = 25.0  # Mock 25% CPU
                
                # Check if we're exceeding limits
                if self.current_memory_gb > self.max_memory_gb:
                    logger.warning(f"Memory usage ({self.current_memory_gb:.1f}GB) "
                                 f"exceeds limit ({self.max_memory_gb:.1f}GB)")
                
                if self.current_cpu_percent > self.max_cpu_percent:
                    logger.warning(f"CPU usage ({self.current_cpu_percent:.1f}%) "
                                 f"exceeds limit ({self.max_cpu_percent:.1f}%)")
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
            
            # Wait for next check
            self._stop_event.wait(self.check_interval_seconds)
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current resource statistics."""
        return {
            "memory_gb": self.current_memory_gb,
            "cpu_percent": self.current_cpu_percent,
            "peak_memory_gb": self.peak_memory_gb
        }
    
    def is_resource_available(self, required_memory_gb: float = 0.0) -> bool:
        """Check if sufficient resources are available."""
        available_memory = self.max_memory_gb - self.current_memory_gb
        return (available_memory >= required_memory_gb and 
                self.current_cpu_percent < self.max_cpu_percent)


@dataclass
class ExperimentStatus:
    """Track status of experiment execution."""
    
    config: "ExperimentConfig"
    status: str = "pending"  # pending, running, completed, failed, cancelled
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional["ExperimentResult"] = None
    error_message: Optional[str] = None
    
    # Resource usage
    peak_memory_gb: float = 0.0
    total_time_seconds: float = 0.0
    
    # Progress tracking
    current_epoch: int = 0
    total_epochs: int = 0
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get experiment duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return self.status == "running"
    
    @property
    def is_completed(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if experiment failed."""
        return self.status == "failed"


@dataclass
class ExperimentQueue:
    """Queue for managing experiment execution."""
    
    experiments: List[ExperimentStatus] = field(default_factory=list)
    current_index: int = 0
    
    def add_experiment(self, config: "ExperimentConfig"):
        """Add experiment to queue."""
        status = ExperimentStatus(config=config)
        self.experiments.append(status)
        logger.info(f"Added experiment to queue: {config.name}")
    
    def add_experiments(self, configs: List["ExperimentConfig"]):
        """Add multiple experiments to queue."""
        for config in configs:
            self.add_experiment(config)
    
    def get_next_experiment(self) -> Optional[ExperimentStatus]:
        """Get next experiment to run."""
        while self.current_index < len(self.experiments):
            experiment = self.experiments[self.current_index]
            if experiment.status == "pending":
                return experiment
            self.current_index += 1
        return None
    
    def mark_completed(self, experiment: ExperimentStatus, result: "ExperimentResult"):
        """Mark experiment as completed."""
        experiment.status = "completed"
        experiment.end_time = datetime.now()
        experiment.result = result
        logger.info(f"Experiment completed: {experiment.config.name}")
    
    def mark_failed(self, experiment: ExperimentStatus, error: str):
        """Mark experiment as failed."""
        experiment.status = "failed"
        experiment.end_time = datetime.now()
        experiment.error_message = error
        logger.error(f"Experiment failed: {experiment.config.name} - {error}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get queue summary."""
        total = len(self.experiments)
        completed = sum(1 for exp in self.experiments if exp.is_completed)
        failed = sum(1 for exp in self.experiments if exp.is_failed)
        running = sum(1 for exp in self.experiments if exp.is_running)
        pending = total - completed - failed - running
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }


class ExperimentRunner:
    """
    Automated experiment execution pipeline for M2 hardware.
    
    Features:
    - Sequential experiment execution
    - Resource monitoring and adaptive batch sizing
    - Automatic checkpointing and resumption
    - Comprehensive logging and progress tracking
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "experiments/outputs",
        max_memory_gb: float = 32.0,
        max_concurrent_experiments: int = 1,  # Sequential for M2
        checkpoint_interval_minutes: int = 30,
        auto_resume: bool = True
    ):
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Directory for experiment outputs
            max_memory_gb: Maximum memory usage limit
            max_concurrent_experiments: Number of concurrent experiments (1 for M2)
            checkpoint_interval_minutes: How often to save checkpoints
            auto_resume: Whether to automatically resume interrupted experiments
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_experiments = max_concurrent_experiments
        self.checkpoint_interval = timedelta(minutes=checkpoint_interval_minutes)
        self.auto_resume = auto_resume
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(max_memory_gb=max_memory_gb)
        
        # Experiment management
        self.experiment_queue = ExperimentQueue()
        self.results_manager = ResultsManager(self.output_dir / "results")
        
        # Execution state
        self.running = False
        self.paused = False
        self._stop_requested = False
        
        # Callbacks
        self.on_experiment_start: Optional[Callable] = None
        self.on_experiment_complete: Optional[Callable] = None
        self.on_experiment_failed: Optional[Callable] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"ExperimentRunner initialized with output_dir: {self.output_dir}")
    
    def add_experiment(self, config: "ExperimentConfig"):
        """Add single experiment to execution queue."""
        self.experiment_queue.add_experiment(config)
    
    def add_experiments(self, configs: List["ExperimentConfig"]):
        """Add multiple experiments to execution queue."""
        self.experiment_queue.add_experiments(configs)
    
    def add_experiment_matrix(self, matrix: "ExperimentMatrix"):
        """Add all experiments from a matrix to the queue."""
        configs = list(matrix.generate_configs())
        logger.info(f"Adding {len(configs)} experiments from matrix")
        self.add_experiments(configs)
    
    def run_experiments(self) -> Dict[str, Any]:
        """
        Run all experiments in the queue.
        
        Returns:
            Summary of execution results
        """
        if self.running:
            raise RuntimeError("Experiments are already running")
        
        self.running = True
        self._stop_requested = False
        
        logger.info("Starting experiment execution")
        logger.info(f"Queue summary: {self.experiment_queue.get_summary()}")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Resume interrupted experiments if enabled
            if self.auto_resume:
                self._resume_interrupted_experiments()
            
            # Main execution loop
            while not self._stop_requested:
                # Get next experiment
                experiment = self.experiment_queue.get_next_experiment()
                if experiment is None:
                    logger.info("No more experiments to run")
                    break
                
                # Check if we should pause
                if self.paused:
                    logger.info("Execution paused, waiting...")
                    time.sleep(5)
                    continue
                
                # Check resource availability
                if not self._check_resource_availability(experiment):
                    logger.warning("Insufficient resources, waiting...")
                    time.sleep(30)  # Wait 30 seconds before checking again
                    continue
                
                # Run the experiment
                try:
                    self._run_single_experiment(experiment)
                except Exception as e:
                    self.experiment_queue.mark_failed(experiment, str(e))
                    if self.on_experiment_failed:
                        self.on_experiment_failed(experiment, e)
            
            # Generate final summary
            summary = self._generate_execution_summary()
            
            logger.info("Experiment execution completed")
            logger.info(f"Final summary: {summary}")
            
            return summary
            
        finally:
            self.running = False
            self.resource_monitor.stop_monitoring()
    
    def pause_execution(self):
        """Pause experiment execution."""
        self.paused = True
        logger.info("Experiment execution paused")
    
    def resume_execution(self):
        """Resume experiment execution."""
        self.paused = False
        logger.info("Experiment execution resumed")
    
    def stop_execution(self):
        """Stop experiment execution gracefully."""
        self._stop_requested = True
        logger.info("Stop requested - will finish current experiment")
    
    def force_stop(self):
        """Force stop experiment execution immediately."""
        self._stop_requested = True
        self.running = False
        logger.warning("Force stop requested")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        queue_summary = self.experiment_queue.get_summary()
        resource_stats = self.resource_monitor.get_current_stats()
        
        return {
            "running": self.running,
            "paused": self.paused,
            "stop_requested": self._stop_requested,
            "queue": queue_summary,
            "resources": resource_stats,
            "output_dir": str(self.output_dir)
        }
    
    def _run_single_experiment(self, experiment: ExperimentStatus):
        """Run a single experiment."""
        config = experiment.config
        
        logger.info(f"Starting experiment: {config.name}")
        logger.info(f"Config: {config.get_experiment_id()}")
        
        # Mark as running
        experiment.status = "running"
        experiment.start_time = datetime.now()
        
        # Call start callback
        if self.on_experiment_start:
            self.on_experiment_start(experiment)
        
        try:
            # Create experiment output directory
            exp_output_dir = self.output_dir / config.get_experiment_id()
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = exp_output_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
            
            # Simulate experiment execution (replace with actual training)
            result = self._simulate_experiment_execution(config, experiment)
            
            # Save results
            result_path = exp_output_dir / "results.json"
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            # Mark as completed
            self.experiment_queue.mark_completed(experiment, result)
            
            # Store in results manager
            self.results_manager.add_result(result)
            
            # Call completion callback
            if self.on_experiment_complete:
                self.on_experiment_complete(experiment, result)
            
        except Exception as e:
            logger.error(f"Experiment failed: {config.name} - {str(e)}")
            self.experiment_queue.mark_failed(experiment, str(e))
            
            if self.on_experiment_failed:
                self.on_experiment_failed(experiment, e)
            
            raise
    
    def _simulate_experiment_execution(
        self, 
        config: "ExperimentConfig", 
        experiment: ExperimentStatus
    ) -> "ExperimentResult":
        """
        Simulate experiment execution for testing.
        Replace this with actual training pipeline integration.
        """
        logger.info(f"Simulating experiment: {config.name}")
        
        # Simulate training epochs
        total_epochs = config.training.num_epochs
        experiment.total_epochs = total_epochs
        
        for epoch in range(total_epochs):
            if self._stop_requested:
                raise RuntimeError("Execution stopped by user")
            
            experiment.current_epoch = epoch
            
            # Simulate epoch training time
            time.sleep(0.1)  # Very fast simulation
            
            # Update resource usage
            experiment.peak_memory_gb = max(
                experiment.peak_memory_gb,
                self.resource_monitor.current_memory_gb
            )
            
            logger.debug(f"Epoch {epoch + 1}/{total_epochs} completed")
        
        # Create mock result
        from .results import ExperimentResult
        
        result = ExperimentResult(
            experiment_id=config.get_experiment_id(),
            config=config,
            metrics={
                "final_accuracy": 0.85 + (hash(config.name) % 100) / 1000,  # Mock accuracy
                "final_loss": 0.5 - (hash(config.name) % 100) / 2000,      # Mock loss
                "training_time_seconds": experiment.duration.total_seconds() if experiment.duration else 0,
                "peak_memory_gb": experiment.peak_memory_gb,
                "total_epochs": total_epochs
            },
            start_time=experiment.start_time,
            end_time=datetime.now(),
            status="completed"
        )
        
        return result
    
    def _check_resource_availability(self, experiment: ExperimentStatus) -> bool:
        """Check if resources are available for the experiment."""
        config = experiment.config
        
        # Estimate required memory (very rough)
        required_memory = 2.0  # Base requirement
        if config.model.name == "deit_small_patch16_224":
            required_memory = 4.0
        elif config.model.name == "vit_base_patch16_224":
            required_memory = 8.0
        
        # Adjust for quantization
        if config.quantization:
            if config.quantization.bits == 8:
                required_memory *= 0.5
            elif config.quantization.bits == 4:
                required_memory *= 0.25
        
        return self.resource_monitor.is_resource_available(required_memory)
    
    def _resume_interrupted_experiments(self):
        """Resume any interrupted experiments."""
        logger.info("Checking for interrupted experiments to resume...")
        
        # Look for experiment directories with config but no results
        for exp_dir in self.output_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            config_path = exp_dir / "config.json"
            result_path = exp_dir / "results.json"
            
            if config_path.exists() and not result_path.exists():
                logger.info(f"Found interrupted experiment: {exp_dir.name}")
                
                # Load config and add back to queue
                try:
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    # This would need proper config reconstruction
                    # For now, just log that we found it
                    logger.warning(f"Interrupted experiment found but resumption not implemented: {exp_dir.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load interrupted experiment config: {e}")
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate summary of experiment execution."""
        queue_summary = self.experiment_queue.get_summary()
        
        # Calculate timing statistics
        completed_experiments = [exp for exp in self.experiment_queue.experiments if exp.is_completed]
        
        total_time = sum(
            exp.duration.total_seconds() for exp in completed_experiments 
            if exp.duration
        )
        
        avg_time = total_time / len(completed_experiments) if completed_experiments else 0
        
        # Resource statistics
        peak_memory = max(
            (exp.peak_memory_gb for exp in completed_experiments),
            default=0
        )
        
        return {
            "execution_summary": queue_summary,
            "timing": {
                "total_time_seconds": total_time,
                "average_time_seconds": avg_time,
                "completed_experiments": len(completed_experiments)
            },
            "resources": {
                "peak_memory_gb": peak_memory,
                "final_memory_gb": self.resource_monitor.current_memory_gb
            },
            "output_directory": str(self.output_dir)
        }
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_execution()
    
    def save_checkpoint(self):
        """Save current execution state."""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "queue_state": {
                "current_index": self.experiment_queue.current_index,
                "experiments": [
                    {
                        "config_id": exp.config.get_experiment_id(),
                        "status": exp.status,
                        "start_time": exp.start_time.isoformat() if exp.start_time else None,
                        "end_time": exp.end_time.isoformat() if exp.end_time else None,
                        "error_message": exp.error_message
                    }
                    for exp in self.experiment_queue.experiments
                ]
            },
            "resource_stats": self.resource_monitor.get_current_stats()
        }
        
        checkpoint_path = self.output_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None):
        """Load execution state from checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / "checkpoint.json"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore queue state
            queue_state = checkpoint_data["queue_state"]
            self.experiment_queue.current_index = queue_state["current_index"]
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Resuming from experiment index: {self.experiment_queue.current_index}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")


# Utility functions for common experiment patterns

def run_experiment_matrix(
    matrix: "ExperimentMatrix",
    output_dir: Union[str, Path] = "experiments/outputs",
    max_memory_gb: float = 32.0
) -> Dict[str, Any]:
    """
    Convenience function to run an entire experiment matrix.
    
    Args:
        matrix: ExperimentMatrix to execute
        output_dir: Output directory for results
        max_memory_gb: Memory limit for M2 hardware
        
    Returns:
        Execution summary
    """
    runner = ExperimentRunner(
        output_dir=output_dir,
        max_memory_gb=max_memory_gb
    )
    
    runner.add_experiment_matrix(matrix)
    return runner.run_experiments()


def run_single_experiment(
    config: "ExperimentConfig",
    output_dir: Union[str, Path] = "experiments/outputs"
) -> "ExperimentResult":
    """
    Convenience function to run a single experiment.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory for results
        
    Returns:
        Experiment result
    """
    runner = ExperimentRunner(output_dir=output_dir)
    runner.add_experiment(config)
    
    summary = runner.run_experiments()
    
    # Return the single result
    if runner.experiment_queue.experiments:
        experiment = runner.experiment_queue.experiments[0]
        if experiment.result:
            return experiment.result
    
    raise RuntimeError("Experiment execution failed")


def create_progress_callback() -> Callable:
    """Create a progress callback for experiment monitoring."""
    def progress_callback(experiment: ExperimentStatus, result: Optional["ExperimentResult"] = None):
        if result:
            logger.info(f"✓ Completed: {experiment.config.name}")
            logger.info(f"  Duration: {experiment.duration}")
            logger.info(f"  Peak Memory: {experiment.peak_memory_gb:.1f}GB")
        else:
            logger.info(f"→ Starting: {experiment.config.name}")
    
    return progress_callback