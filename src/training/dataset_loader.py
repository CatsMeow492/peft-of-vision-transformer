"""
Dataset loading and preprocessing for Vision Transformer PEFT training.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
else:
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, datasets
        from PIL import Image
    except ImportError:
        torch = None
        Dataset = None
        DataLoader = None
        transforms = None
        datasets = None
        Image = None

logger = logging.getLogger(__name__)


class TinyImageNetDataset(Dataset):
    """
    Custom dataset class for TinyImageNet.
    
    TinyImageNet is a subset of ImageNet with 200 classes, 64x64 images.
    Each class has 500 training images, 50 validation images, and 50 test images.
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        """
        Initialize TinyImageNet dataset.
        
        Args:
            root: Root directory for dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to images
            download: Whether to download dataset if not found
        """
        if torch is None or Image is None:
            raise RuntimeError("PyTorch and PIL not available")
        
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Download dataset if needed
        if download and not self._check_dataset_exists():
            self._download_dataset()
        
        # Load class information
        self.classes, self.class_to_idx = self._load_classes()
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        logger.info(f"TinyImageNet {split} dataset loaded: {len(self.samples)} samples, {len(self.classes)} classes")
    
    def _check_dataset_exists(self) -> bool:
        """Check if TinyImageNet dataset exists."""
        required_dirs = ["train", "val", "test"]
        return all((self.root / "tiny-imagenet-200" / d).exists() for d in required_dirs)
    
    def _download_dataset(self):
        """Download TinyImageNet dataset."""
        import urllib.request
        import zipfile
        
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = self.root / "tiny-imagenet-200.zip"
        
        logger.info("Downloading TinyImageNet dataset...")
        self.root.mkdir(parents=True, exist_ok=True)
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            
            # Remove zip file
            zip_path.unlink()
            
            logger.info("TinyImageNet dataset downloaded and extracted")
            
        except Exception as e:
            logger.error(f"Failed to download TinyImageNet: {str(e)}")
            raise RuntimeError(f"Dataset download failed: {str(e)}") from e
    
    def _load_classes(self) -> Tuple[list, dict]:
        """Load class names and create class-to-index mapping."""
        dataset_root = self.root / "tiny-imagenet-200"
        
        # Read class names from wnids.txt
        wnids_path = dataset_root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Class file not found: {wnids_path}")
        
        with open(wnids_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        return classes, class_to_idx
    
    def _load_samples(self) -> list:
        """Load image paths and labels for the specified split."""
        dataset_root = self.root / "tiny-imagenet-200"
        samples = []
        
        if self.split == "train":
            # Training images are organized in class subdirectories
            train_dir = dataset_root / "train"
            for class_name in self.classes:
                class_dir = train_dir / class_name / "images"
                if class_dir.exists():
                    for img_path in class_dir.glob("*.JPEG"):
                        samples.append((str(img_path), self.class_to_idx[class_name]))
        
        elif self.split == "val":
            # Validation images are in a single directory with annotations
            val_dir = dataset_root / "val"
            val_annotations = val_dir / "val_annotations.txt"
            
            if val_annotations.exists():
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            class_name = parts[1]
                            img_path = val_dir / "images" / img_name
                            if img_path.exists() and class_name in self.class_to_idx:
                                samples.append((str(img_path), self.class_to_idx[class_name]))
        
        elif self.split == "test":
            # Test images (no labels available)
            test_dir = dataset_root / "test" / "images"
            if test_dir.exists():
                for img_path in test_dir.glob("*.JPEG"):
                    samples.append((str(img_path), -1))  # No label for test
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {str(e)}")
            # Return a black image as fallback
            image = Image.new('RGB', (64, 64), (0, 0, 0))
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


class DatasetManager:
    """
    Manager for loading and preprocessing datasets for Vision Transformer training.
    
    Supports CIFAR-10, CIFAR-100, and TinyImageNet with ViT-compatible preprocessing.
    """
    
    SUPPORTED_DATASETS = {
        "cifar10": {
            "num_classes": 10,
            "image_size": (32, 32),
            "channels": 3,
            "description": "CIFAR-10 dataset with 10 classes"
        },
        "cifar100": {
            "num_classes": 100,
            "image_size": (32, 32),
            "channels": 3,
            "description": "CIFAR-100 dataset with 100 classes"
        },
        "tiny_imagenet": {
            "num_classes": 200,
            "image_size": (64, 64),
            "channels": 3,
            "description": "TinyImageNet dataset with 200 classes"
        }
    }
    
    def __init__(self, data_root: str = "./data"):
        """
        Initialize dataset manager.
        
        Args:
            data_root: Root directory for storing datasets
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded datasets
        self._dataset_cache: Dict[str, Dict[str, Dataset]] = {}
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
            
        Raises:
            ValueError: If dataset is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}"
            )
        
        return self.SUPPORTED_DATASETS[dataset_name].copy()
    
    def create_transforms(
        self,
        dataset_name: str,
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
        augmentation_strength: str = "medium"
    ) -> "transforms.Compose":
        """
        Create transforms for dataset preprocessing.
        
        Args:
            dataset_name: Name of the dataset
            target_size: Target image size for ViT (default: 224x224)
            is_training: Whether transforms are for training (includes augmentation)
            augmentation_strength: Strength of data augmentation ("light", "medium", "strong")
            
        Returns:
            Composed transforms
        """
        if transforms is None:
            raise RuntimeError("torchvision not available")
        
        dataset_info = self.get_dataset_info(dataset_name)
        original_size = dataset_info["image_size"]
        
        transform_list = []
        
        # Resize to target size if needed
        if original_size != target_size:
            transform_list.append(transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC))
        
        # Training augmentations
        if is_training:
            if augmentation_strength in ["medium", "strong"]:
                # Random horizontal flip
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            if augmentation_strength == "strong":
                # Additional augmentations for strong setting
                transform_list.extend([
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
                ])
            elif augmentation_strength == "medium":
                # Moderate augmentations
                transform_list.extend([
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
                ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        target_size: Tuple[int, int] = (224, 224),
        augmentation_strength: str = "medium",
        download: bool = True
    ) -> "Dataset":
        """
        Load a dataset with appropriate preprocessing.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split ("train", "val", "test")
            target_size: Target image size for ViT
            augmentation_strength: Strength of data augmentation
            download: Whether to download dataset if not found
            
        Returns:
            Loaded dataset
            
        Raises:
            ValueError: If dataset or split is not supported
            RuntimeError: If dataset loading fails
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported split: {split}")
        
        # Check cache
        cache_key = f"{dataset_name}_{split}_{target_size}_{augmentation_strength}"
        if cache_key in self._dataset_cache:
            logger.info(f"Returning cached dataset: {cache_key}")
            return self._dataset_cache[cache_key]
        
        try:
            # Create transforms
            is_training = (split == "train")
            transform = self.create_transforms(
                dataset_name=dataset_name,
                target_size=target_size,
                is_training=is_training,
                augmentation_strength=augmentation_strength
            )
            
            # Load dataset
            if dataset_name == "cifar10":
                dataset = datasets.CIFAR10(
                    root=str(self.data_root),
                    train=(split == "train"),
                    transform=transform,
                    download=download
                )
            elif dataset_name == "cifar100":
                dataset = datasets.CIFAR100(
                    root=str(self.data_root),
                    train=(split == "train"),
                    transform=transform,
                    download=download
                )
            elif dataset_name == "tiny_imagenet":
                dataset = TinyImageNetDataset(
                    root=str(self.data_root),
                    split=split,
                    transform=transform,
                    download=download
                )
            else:
                raise ValueError(f"Dataset loading not implemented: {dataset_name}")
            
            # Cache the dataset
            self._dataset_cache[cache_key] = dataset
            
            logger.info(f"Dataset loaded: {dataset_name} {split} with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name} {split}: {str(e)}")
            raise RuntimeError(f"Dataset loading failed: {str(e)}") from e
    
    def create_dataloaders(
        self,
        dataset_name: str,
        batch_size: int = 32,
        target_size: Tuple[int, int] = (224, 224),
        augmentation_strength: str = "medium",
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: float = 0.9,
        seed: int = 42
    ) -> Dict[str, "DataLoader"]:
        """
        Create train/validation/test dataloaders for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            batch_size: Batch size for dataloaders
            target_size: Target image size for ViT
            augmentation_strength: Strength of data augmentation
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            train_val_split: Ratio for train/validation split (only for datasets without separate val set)
            seed: Random seed for reproducible splits
            
        Returns:
            Dictionary with 'train', 'val', and optionally 'test' dataloaders
        """
        if DataLoader is None:
            raise RuntimeError("PyTorch DataLoader not available")
        
        dataloaders = {}
        
        try:
            # Set random seed for reproducible splits
            torch.manual_seed(seed)
            
            # Load datasets based on availability
            if dataset_name in ["cifar10", "cifar100"]:
                # CIFAR datasets have train/test splits, need to create validation split
                train_dataset = self.load_dataset(
                    dataset_name=dataset_name,
                    split="train",
                    target_size=target_size,
                    augmentation_strength=augmentation_strength
                )
                
                test_dataset = self.load_dataset(
                    dataset_name=dataset_name,
                    split="test",
                    target_size=target_size,
                    augmentation_strength="light"  # No augmentation for test
                )
                
                # Split training data into train/val
                train_size = int(train_val_split * len(train_dataset))
                val_size = len(train_dataset) - train_size
                
                train_subset, val_subset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(seed)
                )
                
                # Create dataloaders
                dataloaders["train"] = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True
                )
                
                dataloaders["val"] = DataLoader(
                    val_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False
                )
                
                dataloaders["test"] = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False
                )
            
            elif dataset_name == "tiny_imagenet":
                # TinyImageNet has separate train/val/test splits
                for split in ["train", "val", "test"]:
                    dataset = self.load_dataset(
                        dataset_name=dataset_name,
                        split=split,
                        target_size=target_size,
                        augmentation_strength=augmentation_strength if split == "train" else "light"
                    )
                    
                    dataloaders[split] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=(split == "train"),
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        drop_last=(split == "train")
                    )
            
            # Log dataloader information
            for split, dataloader in dataloaders.items():
                logger.info(f"{split.capitalize()} dataloader: {len(dataloader)} batches, "
                           f"batch_size={batch_size}")
            
            return dataloaders
            
        except Exception as e:
            logger.error(f"Failed to create dataloaders for {dataset_name}: {str(e)}")
            raise RuntimeError(f"Dataloader creation failed: {str(e)}") from e
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get statistics about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            # Load a small sample to compute statistics
            dataset = self.load_dataset(dataset_name, split="train", download=False)
            
            # Sample a subset for statistics computation
            sample_size = min(1000, len(dataset))
            indices = torch.randperm(len(dataset))[:sample_size]
            
            pixel_values = []
            labels = []
            
            for idx in indices:
                image, label = dataset[idx]
                pixel_values.append(image)
                labels.append(label)
            
            # Stack tensors
            pixel_values = torch.stack(pixel_values)
            labels = torch.tensor(labels)
            
            # Compute statistics
            stats = {
                "dataset_name": dataset_name,
                "total_samples": len(dataset),
                "sample_size_for_stats": sample_size,
                "num_classes": len(torch.unique(labels)),
                "image_shape": list(pixel_values.shape[1:]),
                "pixel_mean": pixel_values.mean(dim=[0, 2, 3]).tolist(),
                "pixel_std": pixel_values.std(dim=[0, 2, 3]).tolist(),
                "pixel_min": pixel_values.min().item(),
                "pixel_max": pixel_values.max().item(),
                "class_distribution": torch.bincount(labels).tolist()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute statistics for {dataset_name}: {str(e)}")
            return {"error": str(e)}
    
    def create_custom_collate_fn(self) -> Callable:
        """
        Create a custom collate function for Vision Transformer training.
        
        Returns:
            Collate function that creates batches with 'pixel_values' and 'labels' keys
        """
        def collate_fn(batch):
            """Custom collate function for ViT training."""
            images, labels = zip(*batch)
            
            return {
                "pixel_values": torch.stack(images),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        
        return collate_fn
    
    def clear_cache(self):
        """Clear the dataset cache to free memory."""
        self._dataset_cache.clear()
        logger.info("Dataset cache cleared")
    
    def list_supported_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all supported datasets.
        
        Returns:
            Dictionary of supported datasets and their configurations
        """
        return self.SUPPORTED_DATASETS.copy()


def create_memory_efficient_dataloader(
    dataset: "Dataset",
    batch_size: int,
    max_memory_gb: float = 8.0,
    **dataloader_kwargs
) -> "DataLoader":
    """
    Create a memory-efficient dataloader that adapts batch size based on available memory.
    
    Args:
        dataset: Dataset to create dataloader for
        batch_size: Desired batch size
        max_memory_gb: Maximum memory to use in GB
        **dataloader_kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader with potentially adjusted batch size
    """
    if DataLoader is None:
        raise RuntimeError("PyTorch DataLoader not available")
    
    try:
        # Estimate memory usage per sample
        sample_image, _ = dataset[0]
        if hasattr(sample_image, 'numel') and hasattr(sample_image, 'element_size'):
            bytes_per_sample = sample_image.numel() * sample_image.element_size()
            gb_per_sample = bytes_per_sample / (1024 ** 3)
            
            # Calculate maximum batch size based on memory constraint
            max_batch_size = int(max_memory_gb / gb_per_sample)
            
            # Use the smaller of desired batch size and memory-constrained batch size
            adjusted_batch_size = min(batch_size, max_batch_size)
            
            if adjusted_batch_size < batch_size:
                logger.warning(f"Batch size reduced from {batch_size} to {adjusted_batch_size} "
                             f"due to memory constraints ({max_memory_gb}GB limit)")
            
            return DataLoader(dataset, batch_size=adjusted_batch_size, **dataloader_kwargs)
        else:
            # Fallback to original batch size if memory estimation fails
            logger.warning("Could not estimate memory usage, using original batch size")
            return DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)
            
    except Exception as e:
        logger.warning(f"Memory estimation failed: {str(e)}, using original batch size")
        return DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)