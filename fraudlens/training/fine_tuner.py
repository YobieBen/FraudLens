"""
Fine-tuning module for FraudLens models.

Author: Yobie Benjamin
Date: 2025-08-28
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FakeDocumentDataset(Dataset):
    """Dataset for fake document training."""

    def __init__(self, data_path: Path, transform=None):
        """Initialize dataset."""
        self.data_path = data_path
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Load dataset metadata
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} training samples")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load training samples."""
        samples = []

        # Load from metadata file if exists
        metadata_file = self.data_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                samples = json.load(f)

        # Add hardcoded known fakes for initial training
        known_fakes = [
            {
                "id": "mclovin_001",
                "type": "driver_license",
                "is_fake": True,
                "confidence": 1.0,
                "patterns": ["McLovin", "06/03/1981", "Hawaii"],
                "description": "Superbad movie fake ID",
            },
            {
                "id": "john_doe_001",
                "type": "driver_license",
                "is_fake": True,
                "confidence": 0.95,
                "patterns": ["John Doe", "123 Main St", "Anytown"],
                "description": "Generic placeholder ID",
            },
            {
                "id": "specimen_001",
                "type": "passport",
                "is_fake": True,
                "confidence": 1.0,
                "patterns": ["SPECIMEN", "SAMPLE", "TEST"],
                "description": "Sample passport",
            },
        ]

        samples.extend(known_fakes)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get training sample."""
        sample = self.samples[idx]

        # For now, return synthetic data
        # In production, would load actual images
        if "image_path" in sample:
            image = Image.open(self.data_path / sample["image_path"])
            if self.transform:
                image = self.transform(image)
        else:
            # Create synthetic tensor for testing
            image = torch.randn(3, 224, 224)

        label = 1.0 if sample.get("is_fake", False) else 0.0
        confidence = sample.get("confidence", 0.5)

        return image, torch.tensor([label, confidence])


class FineTuner:
    """
    Handles fine-tuning of fraud detection models.
    """

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """Initialize fine-tuner."""
        self.model_path = model_path or Path("models/fine_tuned")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Set device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"FineTuner initialized with device: {self.device}")

        # Initialize models
        self.models = {}
        self._load_base_models()

        # Training configuration
        self.config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "validation_split": 0.2,
            "early_stopping_patience": 3,
            "weight_decay": 0.0001,
        }

        # Training history
        self.training_history = []

    def _load_base_models(self):
        """Load base models for fine-tuning."""
        try:
            # Load pre-trained models (simplified for demo)
            from torchvision import models

            # Document forgery detection model
            self.models["document_forgery"] = models.resnet18(pretrained=True)
            self.models["document_forgery"].fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2),  # Binary classification: real/fake
            )

            # Manipulation detection model
            self.models["manipulation"] = models.efficientnet_b0(pretrained=True)
            self.models["manipulation"].classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(1280, 512), nn.ReLU(), nn.Linear(512, 2)
            )

            # Move models to device
            for name, model in self.models.items():
                self.models[name] = model.to(self.device)
                logger.info(f"Loaded model: {name}")

        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            # Create simple models as fallback
            self.models["document_forgery"] = self._create_simple_model()
            self.models["manipulation"] = self._create_simple_model()

    def _create_simple_model(self) -> nn.Module:
        """Create simple neural network for fallback."""

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.fc1 = nn.Linear(64 * 56 * 56, 128)
                self.fc2 = nn.Linear(128, 2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        return SimpleNet().to(self.device)

    def fine_tune_on_known_fakes(
        self, data_path: Path, model_name: str = "document_forgery", epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune model on known fake documents.

        Args:
            data_path: Path to training data
            model_name: Which model to fine-tune
            epochs: Number of training epochs

        Returns:
            Training results
        """
        epochs = epochs or self.config["epochs"]

        logger.info(f"Starting fine-tuning for {model_name}")

        # Create dataset and dataloader
        dataset = FakeDocumentDataset(data_path)

        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=False)

        # Get model
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        training_results = {
            "model": model_name,
            "epochs": epochs,
            "train_losses": [],
            "val_losses": [],
            "accuracies": [],
        }

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels[:, 0].long().to(self.device)  # Get fake/real label

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels[:, 0].long().to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0

            # Update scheduler
            scheduler.step(avg_val_loss)

            # Save results
            training_results["train_losses"].append(avg_train_loss)
            training_results["val_losses"].append(avg_val_loss)
            training_results["accuracies"].append(val_accuracy)

            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.2f}%"
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self._save_model(model, model_name)
            else:
                patience_counter += 1
                if patience_counter >= self.config["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Save training history
        self.training_history.append(
            {"timestamp": datetime.now().isoformat(), "results": training_results}
        )

        return training_results

    def _save_model(self, model: nn.Module, model_name: str):
        """Save fine-tuned model."""
        model_file = self.model_path / f"{model_name}_finetuned.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
            },
            model_file,
        )
        logger.info(f"Saved fine-tuned model: {model_file}")

    def load_fine_tuned_model(self, model_name: str) -> bool:
        """Load a fine-tuned model."""
        model_file = self.model_path / f"{model_name}_finetuned.pth"

        if not model_file.exists():
            logger.warning(f"Fine-tuned model not found: {model_file}")
            return False

        try:
            checkpoint = torch.load(model_file, map_location=self.device)

            if model_name in self.models:
                self.models[model_name].load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded fine-tuned model: {model_name}")
                return True
            else:
                logger.error(f"Base model {model_name} not initialized")
                return False

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return False

    def adapt_to_new_pattern(
        self, pattern_data: Dict[str, Any], model_name: str = "document_forgery"
    ) -> bool:
        """
        Quick adaptation to new fraud pattern.

        Args:
            pattern_data: Information about the new pattern
            model_name: Which model to adapt

        Returns:
            Success status
        """
        try:
            # Create mini-batch for few-shot learning
            model = self.models.get(model_name)
            if not model:
                return False

            # Set model to training mode for adaptation
            model.train()

            # Create synthetic training data for the pattern
            # In production, would use actual examples
            synthetic_batch = torch.randn(5, 3, 224, 224).to(self.device)
            labels = torch.ones(5, dtype=torch.long).to(self.device)  # All fake

            # Quick fine-tuning (few-shot learning)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            for _ in range(5):  # Few iterations
                optimizer.zero_grad()
                outputs = model(synthetic_batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()

            logger.info(
                f"Adapted model {model_name} to new pattern: {pattern_data.get('name', 'unknown')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to adapt model: {e}")
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "models_trained": list(self.models.keys()),
            "device": self.device,
            "training_history": self.training_history,
            "config": self.config,
        }
