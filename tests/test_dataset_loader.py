"""
Tests for dataset loading and preprocessing functionality.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.training.dataset_loader import DatasetManager, TinyImageNetDataset, create_memory_efficient_dataloader


class TestDatasetManager:
    """Test DatasetManager class."""
    
    def test_initialization(self, tmp_path):
        """Test dataset manager initialization."""
        manager = DatasetManager(data_root=str(tmp_path))
        
        assert manager.data_root == tmp_path
        assert tmp_path.exists()
        assert len(manager._dataset_cache) == 0
    
    def test_supported_datasets(self):
        """Test supported datasets information."""
        manager = DatasetManager()
        
        supported = manager.list_supported_datasets()
        
        assert "cifar10" in supported
        assert "cifar100" in supported
        assert "tiny_imagenet" in supported
        
        # Check CIFAR-10 info
        cifar10_info = supported["cifar10"]
        assert cifar10_info["num_classes"] == 10
        assert cifar10_info["image_size"] == (32, 32)
        assert cifar10_info["channels"] == 3
    
    def test_get_dataset_info(self):
        """Test getting dataset information."""
        manager = DatasetManager()
        
        # Test valid dataset
        info = manager.get_dataset_info("cifar10")
        assert info["num_classes"] == 10
        assert info["image_size"] == (32, 32)
        
        # Test invalid dataset
        with pytest.raises(ValueError, match="not supported"):
            manager.get_dataset_info("invalid_dataset")
    
    @patch('src.training.dataset_loader.transforms')
    def test_create_transforms(self, mock_transforms):
        """Test transform creation."""
        manager = DatasetManager()
        
        # Mock transforms
        mock_compose = Mock()
        mock_transforms.Compose.return_value = mock_compose
        mock_transforms.Resize = Mock()
        mock_transforms.RandomHorizontalFlip = Mock()
        mock_transforms.ToTensor = Mock()
        mock_transforms.Normalize = Mock()
        mock_transforms.InterpolationMode.BICUBIC = "bicubic"
        
        # Test training transforms
        transform = manager.create_transforms(
            dataset_name="cifar10",
            target_size=(224, 224),
            is_training=True,
            augmentation_strength="medium"
        )
        
        assert transform == mock_compose
        mock_transforms.Compose.assert_called_once()
    
    def test_custom_collate_fn(self):
        """Test custom collate function."""
        manager = DatasetManager()
        collate_fn = manager.create_custom_collate_fn()
        
        # Create mock batch
        batch = [
            (torch.randn(3, 224, 224), 0),
            (torch.randn(3, 224, 224), 1),
            (torch.randn(3, 224, 224), 2)
        ]
        
        result = collate_fn(batch)
        
        assert "pixel_values" in result
        assert "labels" in result
        assert result["pixel_values"].shape == (3, 3, 224, 224)
        assert result["labels"].shape == (3,)
        assert torch.equal(result["labels"], torch.tensor([0, 1, 2], dtype=torch.long))
    
    def test_cache_functionality(self):
        """Test dataset caching."""
        manager = DatasetManager()
        
        # Initially empty cache
        assert len(manager._dataset_cache) == 0
        
        # Add something to cache (simulate)
        manager._dataset_cache["test_key"] = "test_value"
        assert len(manager._dataset_cache) == 1
        
        # Clear cache
        manager.clear_cache()
        assert len(manager._dataset_cache) == 0


class TestTinyImageNetDataset:
    """Test TinyImageNetDataset class."""
    
    def test_initialization_without_download(self, tmp_path):
        """Test dataset initialization without downloading."""
        # Create mock directory structure
        dataset_dir = tmp_path / "tiny-imagenet-200"
        dataset_dir.mkdir()
        (dataset_dir / "train").mkdir()
        (dataset_dir / "val").mkdir()
        (dataset_dir / "test").mkdir()
        
        # Create mock wnids.txt
        wnids_file = dataset_dir / "wnids.txt"
        wnids_file.write_text("n01443537\nn01629819\nn01641577\n")
        
        # Create mock validation annotations
        val_dir = dataset_dir / "val"
        val_images_dir = val_dir / "images"
        val_images_dir.mkdir()
        
        val_annotations = val_dir / "val_annotations.txt"
        val_annotations.write_text("val_0.JPEG\tn01443537\t0\t0\t62\t62\n")
        
        # Create a mock image file
        mock_image_path = val_images_dir / "val_0.JPEG"
        mock_image_path.write_bytes(b"fake_image_data")
        
        with patch('src.training.dataset_loader.Image') as mock_image:
            mock_img = Mock()
            mock_img.convert.return_value = mock_img
            mock_image.open.return_value = mock_img
            
            # Test dataset creation
            dataset = TinyImageNetDataset(
                root=str(tmp_path),
                split="val",
                download=False
            )
            
            assert len(dataset.classes) == 3
            assert "n01443537" in dataset.classes
            assert dataset.class_to_idx["n01443537"] == 0
    
    def test_class_loading(self, tmp_path):
        """Test class loading functionality."""
        # Create mock directory structure
        dataset_dir = tmp_path / "tiny-imagenet-200"
        dataset_dir.mkdir()
        
        # Create mock wnids.txt
        wnids_file = dataset_dir / "wnids.txt"
        wnids_file.write_text("class1\nclass2\nclass3\n")
        
        dataset = TinyImageNetDataset.__new__(TinyImageNetDataset)
        dataset.root = tmp_path
        
        classes, class_to_idx = dataset._load_classes()
        
        assert classes == ["class1", "class2", "class3"]
        assert class_to_idx == {"class1": 0, "class2": 1, "class3": 2}
    
    def test_missing_wnids_file(self, tmp_path):
        """Test handling of missing wnids.txt file."""
        dataset_dir = tmp_path / "tiny-imagenet-200"
        dataset_dir.mkdir()
        
        dataset = TinyImageNetDataset.__new__(TinyImageNetDataset)
        dataset.root = tmp_path
        
        with pytest.raises(FileNotFoundError, match="Class file not found"):
            dataset._load_classes()


class TestMemoryEfficientDataloader:
    """Test memory-efficient dataloader creation."""
    
    def test_memory_estimation(self):
        """Test memory-based batch size adjustment."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 100
        
        # Create mock sample with known memory usage
        mock_sample = torch.randn(3, 224, 224)  # ~600KB per sample
        mock_dataset.__getitem__.return_value = (mock_sample, 0)
        
        with patch('src.training.dataset_loader.DataLoader') as mock_dataloader_class:
            mock_dataloader = Mock()
            mock_dataloader_class.return_value = mock_dataloader
            
            # Test with memory constraint that should reduce batch size
            result = create_memory_efficient_dataloader(
                dataset=mock_dataset,
                batch_size=1000,  # Very large batch size
                max_memory_gb=0.001,  # Very small memory limit
                shuffle=True
            )
            
            # Should have called DataLoader with reduced batch size
            mock_dataloader_class.assert_called_once()
            call_args = mock_dataloader_class.call_args
            
            # The batch size should be reduced due to memory constraints
            assert call_args[1]["batch_size"] < 1000
            assert call_args[1]["shuffle"] is True
    
    def test_fallback_behavior(self):
        """Test fallback when memory estimation fails."""
        # Create mock dataset that doesn't support memory estimation
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.__getitem__.return_value = ("not_a_tensor", 0)
        
        with patch('src.training.dataset_loader.DataLoader') as mock_dataloader_class:
            mock_dataloader = Mock()
            mock_dataloader_class.return_value = mock_dataloader
            
            result = create_memory_efficient_dataloader(
                dataset=mock_dataset,
                batch_size=32,
                max_memory_gb=8.0
            )
            
            # Should use original batch size as fallback
            mock_dataloader_class.assert_called_once()
            call_args = mock_dataloader_class.call_args
            assert call_args[1]["batch_size"] == 32


if __name__ == "__main__":
    pytest.main([__file__])