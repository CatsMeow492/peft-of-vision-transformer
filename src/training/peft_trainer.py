"""
Base PEFT trainer for Parameter-Efficient Fine-Tuning of Vision Transformers.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    import torch.cuda.amp as amp
else:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch.optim import Optimizer
        from torch.optim.lr_scheduler import _LRScheduler
        import torch.cuda.amp as amp
    except ImportError:
        torch = None
        nn = None
        DataLoader = None
        Optimizer = None
        _LRScheduler = None
        amp = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PEFT training."""
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Optimization settings
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "linear", "constant"
    gradient_clip_norm: Optional[float] = 1.0
    
    # Mixed precision and memory optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing and logging
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    save_total_limit: int = 3
    
    # Output directories
    output_dir: str = "outputs"
    logging_dir: Optional[str] = None
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.001
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        
        # Set default logging directory
        if self.logging_dir is None:
            self.logging_dir = str(Path(self.output_dir) / "logs")


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    epoch: int
    step: int
    train_loss: float
    train_accuracy: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    
    # Timing metrics
    epoch_time: float = 0.0
    step_time: float = 0.0
    
    # Memory metrics
    memory_used_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    
    # Additional metrics
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResults:
    """Container for final training results."""
    
    # Final metrics
    final_train_loss: float
    final_eval_loss: Optional[float] = None
    final_train_accuracy: Optional[float] = None
    final_eval_accuracy: Optional[float] = None
    best_eval_accuracy: Optional[float] = None
    
    # Training statistics
    total_epochs: int = 0
    total_steps: int = 0
    total_training_time: float = 0.0
    
    # Convergence information
    converged: bool = False
    early_stopped: bool = False
    best_checkpoint_path: Optional[str] = None
    
    # Training history
    training_history: List[TrainingMetrics] = field(default_factory=list)


class PEFTTrainer:
    """
    Base trainer for Parameter-Efficient Fine-Tuning of Vision Transformers.
    
    Supports LoRA and other PEFT methods with memory-efficient training,
    gradient accumulation, mixed precision, and comprehensive logging.
    """
    
    def __init__(
        self,
        model,
        config: TrainingConfig,
        train_dataloader: "DataLoader",
        eval_dataloader: Optional["DataLoader"] = None,
        optimizer: Optional["Optimizer"] = None,
        scheduler: Optional["_LRScheduler"] = None,
        compute_metrics: Optional[Callable] = None
    ):
        """
        Initialize PEFT trainer.
        
        Args:
            model: Model to train (should have PEFT adapters applied)
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            optimizer: Optional custom optimizer
            scheduler: Optional learning rate scheduler
            compute_metrics: Optional function to compute additional metrics
        """
        if torch is None:
            raise RuntimeError("PyTorch not available")
        
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = amp.GradScaler() if (amp and config.use_mixed_precision) else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_accuracy = None
        self.early_stopping_counter = 0
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logging_dir = Path(config.logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history: List[TrainingMetrics] = []
        
        logger.info(f"PEFTTrainer initialized with device: {self.device}")
        logger.info(f"Model has {self._count_parameters()['trainable']:,} trainable parameters")
    
    def _create_optimizer(self) -> "Optimizer":
        """Create optimizer based on configuration."""
        # Get trainable parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer.lower() == "adamw":
            from torch.optim import AdamW
            return AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "adam":
            from torch.optim import Adam
            return Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            from torch.optim import SGD
            return SGD(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional["_LRScheduler"]:
        """Create learning rate scheduler based on configuration."""
        if self.config.scheduler.lower() == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs * len(self.train_dataloader),
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler.lower() == "constant":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
            }
        else:
            # For CPU or MPS, return zeros
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0
            }
    
    def train(self) -> TrainingResults:
        """
        Run the complete training loop.
        
        Returns:
            TrainingResults with final metrics and training history
        """
        logger.info("Starting PEFT training")
        logger.info(f"Training configuration: {self.config}")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        start_time = time.time()
        
        try:
            # Training loop
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                train_metrics = self._train_epoch()
                
                # Evaluate if eval dataloader is provided
                eval_metrics = None
                if self.eval_dataloader is not None:
                    eval_metrics = self._evaluate()
                
                # Create epoch metrics
                epoch_metrics = TrainingMetrics(
                    epoch=epoch,
                    step=self.global_step,
                    train_loss=train_metrics["loss"],
                    train_accuracy=train_metrics.get("accuracy"),
                    eval_loss=eval_metrics["loss"] if eval_metrics else None,
                    eval_accuracy=eval_metrics.get("accuracy") if eval_metrics else None,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    grad_norm=train_metrics.get("grad_norm"),
                    epoch_time=train_metrics["epoch_time"],
                    memory_used_mb=self._get_memory_usage()["allocated_mb"]
                )
                
                self.training_history.append(epoch_metrics)
                
                # Log epoch results
                self._log_epoch_results(epoch_metrics)
                
                # Save checkpoint
                if (epoch + 1) % (self.config.save_steps // len(self.train_dataloader)) == 0:
                    self._save_checkpoint(epoch, eval_metrics)
                
                # Check for early stopping
                if self._should_early_stop(eval_metrics):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Calculate final results
            total_time = time.time() - start_time
            final_results = self._create_final_results(total_time)
            
            logger.info("Training completed successfully")
            logger.info(f"Total training time: {total_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}") from e
    
    def _train_epoch(self) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_start_time = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        grad_norms = []
        
        for step, batch in enumerate(self.train_dataloader):
            step_start_time = time.time()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with amp.autocast():
                    loss, logits = self._compute_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                loss, logits = self._compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_norm is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    grad_norms.append(grad_norm.item())
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Calculate accuracy if possible
            if logits is not None and "labels" in batch:
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == batch["labels"]).sum().item()
                total_correct += correct
                total_samples += batch["labels"].size(0)
            
            # Log step results
            if self.global_step % self.config.logging_steps == 0:
                step_time = time.time() - step_start_time
                self._log_step_results(step, loss.item(), step_time)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else None
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else None
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "grad_norm": avg_grad_norm,
            "epoch_time": epoch_time
        }
    
    def _evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.scaler is not None:
                    with amp.autocast():
                        loss, logits = self._compute_loss(batch)
                else:
                    loss, logits = self._compute_loss(batch)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if logits is not None and "labels" in batch:
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == batch["labels"]).sum().item()
                    total_correct += correct
                    total_samples += batch["labels"].size(0)
        
        avg_loss = total_loss / len(self.eval_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else None
        
        # Compute additional metrics if provided
        additional_metrics = {}
        if self.compute_metrics is not None:
            try:
                additional_metrics = self.compute_metrics(self.model, self.eval_dataloader)
            except Exception as e:
                logger.warning(f"Failed to compute additional metrics: {str(e)}")
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            **additional_metrics
        }
    
    def _compute_loss(self, batch: Dict[str, "torch.Tensor"]) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Compute loss for a batch.
        
        Args:
            batch: Dictionary containing 'pixel_values' and 'labels'
            
        Returns:
            Tuple of (loss, logits)
        """
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        # Forward pass
        outputs = self.model(pixel_values)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            # Try to get the first tensor output
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        # Compute cross-entropy loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        return loss, logits
    
    def _move_batch_to_device(self, batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Move batch tensors to the appropriate device."""
        return {key: value.to(self.device) for key, value in batch.items()}
    
    def _should_early_stop(self, eval_metrics: Optional[Dict[str, Any]]) -> bool:
        """Check if training should be stopped early."""
        if self.config.early_stopping_patience is None or eval_metrics is None:
            return False
        
        current_accuracy = eval_metrics.get("accuracy")
        if current_accuracy is None:
            return False
        
        # Update best accuracy
        if self.best_eval_accuracy is None or current_accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = current_accuracy
            self.early_stopping_counter = 0
            return False
        
        # Check if improvement is below threshold
        improvement = current_accuracy - self.best_eval_accuracy
        if improvement < self.config.early_stopping_threshold:
            self.early_stopping_counter += 1
        else:
            self.early_stopping_counter = 0
        
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, eval_metrics: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model state
        if hasattr(self.model, 'save_pretrained'):
            # PEFT model
            self.model.save_pretrained(checkpoint_dir)
        else:
            # Regular PyTorch model
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        
        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_eval_accuracy": self.best_eval_accuracy,
            "config": self.config,
            "training_history": self.training_history
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # Save evaluation metrics if available
        if eval_metrics is not None:
            torch.save(eval_metrics, checkpoint_dir / "eval_metrics.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _log_step_results(self, step: int, loss: float, step_time: float):
        """Log results for a training step."""
        memory_stats = self._get_memory_usage()
        lr = self.optimizer.param_groups[0]["lr"]
        
        logger.info(
            f"Step {self.global_step} | Epoch {self.current_epoch} | "
            f"Loss: {loss:.4f} | LR: {lr:.2e} | "
            f"Time: {step_time:.2f}s | Memory: {memory_stats['allocated_mb']:.1f}MB"
        )
    
    def _log_epoch_results(self, metrics: TrainingMetrics):
        """Log results for an epoch."""
        log_msg = (
            f"Epoch {metrics.epoch} completed | "
            f"Train Loss: {metrics.train_loss:.4f}"
        )
        
        if metrics.train_accuracy is not None:
            log_msg += f" | Train Acc: {metrics.train_accuracy:.4f}"
        
        if metrics.eval_loss is not None:
            log_msg += f" | Eval Loss: {metrics.eval_loss:.4f}"
        
        if metrics.eval_accuracy is not None:
            log_msg += f" | Eval Acc: {metrics.eval_accuracy:.4f}"
        
        log_msg += f" | Time: {metrics.epoch_time:.2f}s"
        
        logger.info(log_msg)
    
    def _create_final_results(self, total_time: float) -> TrainingResults:
        """Create final training results."""
        if not self.training_history:
            raise RuntimeError("No training history available")
        
        final_metrics = self.training_history[-1]
        
        # Find best evaluation accuracy
        best_eval_accuracy = None
        if any(m.eval_accuracy is not None for m in self.training_history):
            best_eval_accuracy = max(
                m.eval_accuracy for m in self.training_history 
                if m.eval_accuracy is not None
            )
        
        return TrainingResults(
            final_train_loss=final_metrics.train_loss,
            final_eval_loss=final_metrics.eval_loss,
            final_train_accuracy=final_metrics.train_accuracy,
            final_eval_accuracy=final_metrics.eval_accuracy,
            best_eval_accuracy=best_eval_accuracy,
            total_epochs=self.current_epoch + 1,
            total_steps=self.global_step,
            total_training_time=total_time,
            converged=True,  # Assume converged if training completed
            early_stopped=self.early_stopping_counter >= (self.config.early_stopping_patience or float('inf')),
            training_history=self.training_history.copy()
        )
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load model state
        if hasattr(self.model, 'load_adapter'):
            # PEFT model
            self.model.load_adapter(checkpoint_path)
        else:
            # Regular PyTorch model
            model_path = checkpoint_path / "model.pt"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.device)
            
            self.current_epoch = training_state["epoch"]
            self.global_step = training_state["global_step"]
            self.best_eval_accuracy = training_state["best_eval_accuracy"]
            self.training_history = training_state["training_history"]
            
            # Load optimizer and scheduler states
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            
            if self.scheduler and training_state["scheduler_state_dict"]:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
            
            if self.scaler and training_state["scaler_state_dict"]:
                self.scaler.load_state_dict(training_state["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training process."""
        if not self.training_history:
            return {"status": "not_started"}
        
        param_counts = self._count_parameters()
        memory_stats = self._get_memory_usage()
        
        return {
            "status": "completed" if self.current_epoch >= self.config.num_epochs - 1 else "in_progress",
            "current_epoch": self.current_epoch,
            "total_epochs": self.config.num_epochs,
            "global_step": self.global_step,
            "best_eval_accuracy": self.best_eval_accuracy,
            "parameter_counts": param_counts,
            "memory_usage": memory_stats,
            "training_history_length": len(self.training_history)
        }