"""
Deepfake detection using lightweight CNN.

Author: Yobie Benjamin
Date: 2025-08-27 18:48:00 PDT
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from loguru import logger


class DeepfakeDetector:
    """
    Detects deepfake images using lightweight CNN.

    Features:
    - Facial manipulation detection
    - GAN artifact detection
    - Consistency analysis
    - Lightweight model optimized for CPU/Apple Silicon
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_gpu: bool = False,
        threshold: float = 0.5,
    ):
        """
        Initialize deepfake detector.

        Args:
            model_path: Path to model weights
            use_gpu: Use GPU acceleration
            threshold: Detection threshold
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.threshold = threshold
        self.model = None

        # Try to import deep learning libraries
        self.has_torch = self._try_import_torch()
        self.has_onnx = self._try_import_onnx()

        logger.info(
            f"DeepfakeDetector initialized (PyTorch: {self.has_torch}, ONNX: {self.has_onnx})"
        )

    def _try_import_torch(self) -> bool:
        """Try to import PyTorch."""
        try:
            import torch
            import torch.nn as nn

            self.torch = torch
            self.nn = nn
            return True
        except ImportError:
            return False

    def _try_import_onnx(self) -> bool:
        """Try to import ONNX Runtime."""
        try:
            import onnxruntime as ort

            self.ort = ort
            return True
        except ImportError:
            return False

    async def initialize(self) -> None:
        """Load deepfake detection model."""
        if self.has_torch:
            await self._load_torch_model()
        elif self.has_onnx:
            await self._load_onnx_model()
        else:
            logger.warning("No deep learning framework available, using heuristics")

    async def _load_torch_model(self) -> None:
        """Load PyTorch model."""
        # Define lightweight CNN architecture
        nn = self.nn

        class DeepfakeCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    # Conv Block 1
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    # Conv Block 2
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    # Conv Block 3
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )

                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(64, 2),  # Real vs Fake
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        self.model = DeepfakeCNN()

        # Load weights if provided
        if self.model_path and self.model_path.exists():
            self.model.load_state_dict(self.torch.load(self.model_path, map_location="cpu"))

        self.model.eval()

        # Move to GPU if available
        if self.use_gpu and self.torch.cuda.is_available():
            self.model = self.model.cuda()
        elif self.use_gpu and self.torch.backends.mps.is_available():
            self.model = self.model.to("mps")

    async def _load_onnx_model(self) -> None:
        """Load ONNX model."""
        if self.model_path and self.model_path.exists():
            providers = ["CPUExecutionProvider"]
            if self.use_gpu:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

            self.model = self.ort.InferenceSession(str(self.model_path), providers=providers)

    async def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect if image is a deepfake.

        Args:
            image: Input image array

        Returns:
            Detection result
        """
        # Use model if available
        if self.model and self.has_torch:
            return await self._detect_torch(image)
        elif self.model and self.has_onnx:
            return await self._detect_onnx(image)
        else:
            return await self._detect_heuristic(image)

    async def _detect_torch(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect using PyTorch model."""
        # Validate and preprocess image
        if len(image.shape) != 3 or image.shape[2] != 3:
            # Invalid image format, return low confidence result
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "method": "torch",
                "error": f"Invalid image shape: {image.shape}",
            }

        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Resize to model input size
        from PIL import Image

        img = Image.fromarray(image)
        img = img.resize((224, 224))
        image = np.array(img)

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW

        # Add batch dimension
        input_tensor = self.torch.from_numpy(image).unsqueeze(0)

        # Move to device
        if self.use_gpu and self.torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        elif self.use_gpu and self.torch.backends.mps.is_available():
            input_tensor = input_tensor.to("mps")

        # Inference
        with self.torch.no_grad():
            output = self.model(input_tensor)
            probs = self.torch.softmax(output, dim=1)
            fake_prob = probs[0, 1].item()

        is_deepfake = fake_prob > self.threshold

        return {
            "is_deepfake": is_deepfake,
            "confidence": fake_prob if is_deepfake else 1.0 - fake_prob,
            "fake_probability": fake_prob,
            "artifacts": self._detect_artifacts(image) if is_deepfake else [],
        }

    async def _detect_onnx(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect using ONNX model."""
        # Preprocess image
        if image.shape[0] != 224:
            from PIL import Image

            img = Image.fromarray(image)
            img = img.resize((224, 224))
            image = np.array(img)

        # Normalize and prepare input
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        input_data = np.expand_dims(image, 0)

        # Run inference
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        result = self.model.run([output_name], {input_name: input_data})
        probs = self._softmax(result[0][0])
        fake_prob = probs[1]

        is_deepfake = fake_prob > self.threshold

        return {
            "is_deepfake": is_deepfake,
            "confidence": fake_prob if is_deepfake else 1.0 - fake_prob,
            "fake_probability": fake_prob,
            "artifacts": self._detect_artifacts(image) if is_deepfake else [],
        }

    async def _detect_heuristic(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect using heuristic methods."""
        artifacts = []
        suspicious_score = 0.0

        # Check for GAN artifacts
        gan_artifacts = self._detect_gan_artifacts(image)
        if gan_artifacts:
            artifacts.extend(gan_artifacts)
            suspicious_score += 0.3

        # Check for face inconsistencies
        face_issues = self._check_face_consistency(image)
        if face_issues:
            artifacts.extend(face_issues)
            suspicious_score += 0.3

        # Check for frequency domain anomalies
        freq_anomalies = self._check_frequency_domain(image)
        if freq_anomalies:
            artifacts.extend(freq_anomalies)
            suspicious_score += 0.2

        # Check color statistics
        color_anomalies = self._check_color_statistics(image)
        if color_anomalies:
            artifacts.extend(color_anomalies)
            suspicious_score += 0.2

        is_deepfake = suspicious_score > self.threshold

        return {
            "is_deepfake": is_deepfake,
            "confidence": min(suspicious_score, 1.0),
            "fake_probability": suspicious_score,
            "artifacts": artifacts,
        }

    def _detect_artifacts(self, image: np.ndarray) -> list:
        """Detect common deepfake artifacts."""
        artifacts = []

        # Check for boundary artifacts
        if self._has_boundary_artifacts(image):
            artifacts.append("boundary_artifacts")

        # Check for texture inconsistencies
        if self._has_texture_inconsistency(image):
            artifacts.append("texture_inconsistency")

        # Check for lighting inconsistencies
        if self._has_lighting_issues(image):
            artifacts.append("lighting_inconsistency")

        return artifacts

    def _detect_gan_artifacts(self, image: np.ndarray) -> list:
        """Detect GAN-specific artifacts."""
        artifacts = []

        # Check for checkerboard artifacts (common in deconvolution)
        fft = np.fft.fft2(np.mean(image, axis=2))
        fft_mag = np.abs(fft)

        # Look for regular patterns in frequency domain
        peaks = self._find_frequency_peaks(fft_mag)
        if len(peaks) > 5:
            artifacts.append("checkerboard_pattern")

        # Check for mode collapse indicators
        color_variance = np.var(image)
        if color_variance < 100:  # Low variance might indicate mode collapse
            artifacts.append("low_diversity")

        return artifacts

    def _check_face_consistency(self, image: np.ndarray) -> list:
        """Check facial region consistency."""
        issues = []

        # Simple face detection using color-based method
        # In production, would use proper face detection
        skin_mask = self._detect_skin_regions(image)

        if np.any(skin_mask):
            # Check for unnatural skin texture
            skin_pixels = image[skin_mask]
            skin_variance = np.var(skin_pixels)

            if skin_variance < 50:  # Too smooth
                issues.append("unnatural_skin_smoothness")
            elif skin_variance > 5000:  # Too noisy
                issues.append("excessive_skin_noise")

        return issues

    def _check_frequency_domain(self, image: np.ndarray) -> list:
        """Check for frequency domain anomalies."""
        anomalies = []

        # Convert to grayscale
        gray = np.mean(image, axis=2)

        # FFT analysis
        fft = np.fft.fft2(gray)
        fft_mag = np.abs(fft)

        # Check for unusual frequency distribution
        high_freq = np.sum(fft_mag[fft_mag.shape[0] // 2 :])
        low_freq = np.sum(fft_mag[: fft_mag.shape[0] // 2])

        ratio = high_freq / (low_freq + 1e-6)

        if ratio > 0.8:  # Too much high frequency
            anomalies.append("excessive_high_frequency")
        elif ratio < 0.1:  # Too little high frequency
            anomalies.append("missing_high_frequency")

        return anomalies

    def _check_color_statistics(self, image: np.ndarray) -> list:
        """Check color channel statistics."""
        anomalies = []

        # Check channel correlation
        if image.ndim == 3:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

            # Calculate correlations
            rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
            rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
            gb_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]

            # Unusual correlations might indicate manipulation
            if abs(rg_corr - rb_corr) > 0.3:
                anomalies.append("channel_correlation_mismatch")

        return anomalies

    def _has_boundary_artifacts(self, image: np.ndarray) -> bool:
        """Check for artifacts at face boundaries."""
        # Simplified check - look for sharp transitions
        edges = self._detect_edges(image)
        edge_density = np.sum(edges) / edges.size

        return edge_density > 0.15  # High edge density might indicate artifacts

    def _has_texture_inconsistency(self, image: np.ndarray) -> bool:
        """Check for texture inconsistencies."""
        # Calculate texture features using local variance
        from scipy.ndimage import uniform_filter

        gray = np.mean(image, axis=2)
        mean = uniform_filter(gray, size=5)
        variance = uniform_filter(gray**2, size=5) - mean**2

        # Check if variance is too uniform (might indicate generated content)
        variance_std = np.std(variance)

        return variance_std < 5.0

    def _has_lighting_issues(self, image: np.ndarray) -> bool:
        """Check for lighting inconsistencies."""
        # Simple check based on luminance distribution
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

        # Check if lighting is too uniform
        lighting_variance = np.var(luminance)

        return lighting_variance < 100

    def _detect_skin_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect skin regions using color thresholding."""
        # Simple YCrCb-based skin detection
        # In production, would use more sophisticated method

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Define skin color range in RGB
        lower = np.array([20, 40, 50], dtype=np.uint8)
        upper = np.array([255, 200, 180], dtype=np.uint8)

        mask = np.all((image >= lower) & (image <= upper), axis=2)

        return mask

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Sobel filter."""
        from scipy import ndimage

        gray = np.mean(image, axis=2)

        # Sobel edge detection
        sx = ndimage.sobel(gray, axis=0)
        sy = ndimage.sobel(gray, axis=1)
        edges = np.hypot(sx, sy)

        # Threshold
        edges = edges > np.mean(edges)

        return edges

    def _find_frequency_peaks(self, fft_mag: np.ndarray) -> list:
        """Find peaks in frequency domain."""
        from scipy.signal import find_peaks

        # Flatten and find peaks
        flat = fft_mag.flatten()
        peaks, _ = find_peaks(flat, height=np.mean(flat) + 2 * np.std(flat))

        return peaks.tolist()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
